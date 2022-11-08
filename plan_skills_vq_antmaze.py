from comet_ml import Experiment
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.distributions.normal as Normal
from skill_model_vq import SkillModelVectorQuantized
import random
import gym
import d4rl
from mujoco_py import GlfwContext
GlfwContext(offscreen=True)
from math import pi
from search_skills import exhaustive_search
import itertools
from statsmodels.stats.proportion import proportion_confint

device = torch.device('cuda:0')

env = 'antmaze-large-diverse-v0'
#env = 'antmaze-medium-diverse-v0'
#env = 'maze2d-large-v1'

env_name = env
env = gym.make(env)
data = env.get_dataset()

skill_seq_len = 4
H = 20
state_dim = data['observations'].shape[1]
a_dim = data['actions'].shape[1]
h_dim = 256
z_dim = 16
num_embeddings = 16
batch_size = 5000
lr = 1e-4
wd = 0.0
state_dependent_prior = True
state_dec_stop_grad = True
beta = 1.0
alpha = 1.0
ent_pen = 0
max_sig = None
fixed_sig = 0.0#None (Try None for antmaze)
a_dist = 'autoregressive'#'normal'
per_element_sigma = True#False
max_replans = 2000 // H
encoder_type = 'state_action_sequence'
state_decoder_type = 'mlp'
term_state_dependent_prior = False
init_state_dependent = True
random_goal = False#True # determines if we select a goal at random from dataset (random_goal=True) or use pre-set one from environment

filename = 'VQ_model_antmaze-large-diverse-v0_num_embeddings_16_init_state_dep_True_H_20_l2reg_0.0_a_1.0_b_1.0_per_el_sig_True_log_best_a.pth'

PATH = 'checkpoints/'+filename

skill_model = SkillModelVectorQuantized(state_dim, a_dim, z_dim, h_dim, num_embeddings, a_dist=a_dist,state_dec_stop_grad=False,beta=beta,alpha=alpha,max_sig=None,fixed_sig=None,ent_pen=0,encoder_type='state_action_sequence',state_decoder_type=state_decoder_type,init_state_dependent=init_state_dependent,per_element_sigma=per_element_sigma).cuda()

checkpoint = torch.load(PATH)
skill_model.load_state_dict(checkpoint['model_state_dict'])

experiment = Experiment(api_key = 'LVi0h2WLrDaeIC6ZVITGAvzyl', project_name = 'vq_skills')

s0_torch = torch.cat([torch.tensor(env.reset(),dtype=torch.float32).cuda().reshape(1,1,-1) for _ in range(batch_size)])

skill_seq = torch.zeros((1,skill_seq_len,z_dim),device=device)
print('skill_seq.shape: ', skill_seq.shape)
skill_seq.requires_grad = True

idx = [[x for x in range(num_embeddings)]]*skill_seq_len
iterlist = itertools.product(*idx)
idx_list = []
for i in iterlist:
	idx_list.append(i)
idx_list = np.array(idx_list)

def run_skill_seq(skill_seq,env,s0,model):
	state = s0

	pred_states = []
	pred_sigs = []
	states = []
	# plt.figure()
	for i in range(skill_seq.shape[1]):
		# get the skill
		# z = skill_seq[:,i:i+1,:]
		z = skill_seq[:,i:i+1,:]
		skill_seq_states = []
		
		# run skill for H timesteps
		for j in range(H):
			env.render()
			action = model.decoder.ll_policy.numpy_policy(state,z)
			state,_,_,_ = env.step(action)
			skill_seq_states.append(state)
		states.append(skill_seq_states)

	states = np.stack(states)

	return state,states

execute_n_skills = 1

min_dists_list = []
for j in range(1000):
	env.set_target() # this randomizes goal locations between trials, so that we're actualy averaging over the goal distribution
	# otherwise, same goal is kept across resets
	if not random_goal:
		
		goal_state = np.array(env.target_goal)#random.choice(data['observations'])
		print('goal_state: ', goal_state)
	else:
		N = data['observations'].shape[0]
		ind = np.random.randint(low=0,high=N)
		goal_state = data['observations'][ind,:]
	goal_seq = torch.tensor(goal_state, device=device).reshape(1,1,-1)

	state = env.reset()
	goal_loc = goal_state[:2]
	min_dist = 10**10
	for i in range(max_replans):
		if(i%50==0):
			print(i,'/',max_replans)
		s_torch = torch.cat(batch_size*[torch.tensor(state,dtype=torch.float32,device=device).reshape((1,1,-1))]).cuda()
		cost_fn = lambda skill_idx: skill_model.get_expected_cost_vq(s_torch, skill_idx, goal_seq)
		skill_seq = exhaustive_search(skill_model, cost_fn, idx_list, batch_size)
		skill_seq = skill_seq[:execute_n_skills,:]
		skill_seq = skill_seq.unsqueeze(0)	
		state,states = run_skill_seq(skill_seq,env,state,skill_model)

		dists = np.sqrt(np.sum((states[0,:,:2] - goal_loc)**2,axis=-1))

		if np.min(dists) < min_dist:
			min_dist = np.min(dists)

		if min_dist <= 0.5:
			break
		if(i%10==0):
			print(min_dist)
	min_dists_list.append(min_dist)

	p_succ = 0 #Incase need to resume experiments
	p_n_tot = 0 #Incase need to resume experiments
	n_success = np.sum(np.array(min_dists_list) <= 0.5)
	n_tot = len(min_dists_list)

	ci = proportion_confint(n_success+p_succ,n_tot+p_n_tot)
	print('ci: ', ci)
	print('mean: ',(n_success+p_succ)/(n_tot+p_n_tot))
	print('N = ',n_tot+p_n_tot)
	print('n_success = ,',n_success+p_succ)