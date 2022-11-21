from comet_ml import Experiment
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.distributions.normal as Normal
from skill_model_vq import SkillModelVectorQuantized
import d4rl
import gym
from mujoco_py import GlfwContext
GlfwContext(offscreen=True)
from math import pi
from franka_reward_fn import FrankaRewardFn,franka_plan_cost_fn
from search_skills import exhaustive_search, prior_search
from random import sample
import itertools

def run_skill(skill_model,s0,skill,env,H,render,use_epsilon=False):
	try:
		state = s0.flatten().detach().cpu().numpy()
	except:
		state = s0


	if use_epsilon:
		mu_z, sigma_z = skill_model.prior(torch.tensor(s0,device=torch.device('cuda:0'),dtype=torch.float32).unsqueeze(0))
		z = mu_z + sigma_z*skill
	else:
		z = skill
	
	states = [state]
	
	actions = []
	frames = []
	rewards = []
	rew_fn_rewards = []
	for j in range(H): #H-1 if H!=1
		if render:
			frames.append(env.render(mode='rgb_array'))
		action = skill_model.decoder.ll_policy.numpy_policy(state,z)
		actions.append(action)
		state,r,_,_ = env.step(action)
		#print('r: ', r)
		# r_fn = reward_fn.step(state)
		# print('r_fn: ', r_fn)
		
		states.append(state)
		rewards.append(r)

	return state,np.stack(states),np.stack(actions),np.sum(rewards),frames

def run_skill_seq(skill_seq,env,s0,model):
	try:
		state = s0.flatten().detach().cpu().numpy()
	except:
		state = s0

	states = [state]

	actions = []
	rewards = []
	rew_fn_rewards = []
	for i in range(1):
		# get the skill
		# z = skill_seq[:,i:i+1,:]
		z = skill_seq[:,i:i+1,:]
		skill_seq_states = []
		
		# run skill for H timesteps
		for j in range(H):
			env.render()
			action = model.decoder.ll_policy.numpy_policy(state,z)
			state,r,_,_ = env.step(action)
			actions.append(action)
			states.append(state)
			rewards.append(r)

	return state,np.stack(states),np.stack(actions),np.sum(rewards)



if __name__ == '__main__':

	device = torch.device('cuda:0')
	
	env = 'kitchen-partial-v0'

	env_name = env
	env = gym.make(env)
	data = env.get_dataset()

	skill_seq_len = 1#6
	H = 20#30#40
	state_dim = data['observations'].shape[1]
	a_dim = data['actions'].shape[1]
	h_dim = 256
	z_dim = 4
	num_embeddings = 4
	batch_size = 10
	lr = 1e-4
	wd = 0.0
	state_dependent_prior = True
	state_dec_stop_grad = True
	beta = 1.0
	alpha = 1.0
	ent_pen = 0
	max_sig = None
	fixed_sig = 0.0 #(Try None for antmaze)
	a_dist = 'autoregressive'#'normal'
	per_element_sigma = False
	max_replans = 1000 // H
	encoder_type = 'state_action_sequence'
	state_decoder_type = 'mlp'
	term_state_dependent_prior = False
	init_state_dependent = True
	random_goal = False#True # determines if we select a goal at random from dataset (random_goal=True) or use pre-set one from environment


	filename = 'VQ_model_kitchen-partial-v0_num_embeddings_4_init_state_dep_True_zdim_4_H_30_l2reg_0.0_a_1.0_b_1.0_per_el_sig_False_log_best.pth'

	PATH = 'kitchen_partial_checkpoints/'+filename
	
	skill_model = SkillModelVectorQuantized(state_dim, a_dim, z_dim, h_dim, num_embeddings, a_dist=a_dist,state_dec_stop_grad=False,beta=beta,alpha=alpha,max_sig=None,fixed_sig=fixed_sig,ent_pen=0,encoder_type='state_action_sequence',state_decoder_type=state_decoder_type,init_state_dependent=init_state_dependent,per_element_sigma=per_element_sigma).cuda()
		
	checkpoint = torch.load(PATH)
	skill_model.load_state_dict(checkpoint['model_state_dict'])
	

	experiment = Experiment(api_key = 'LVi0h2WLrDaeIC6ZVITGAvzyl', project_name = 'vq_skills')
	experiment.log_parameters({'env_name':env_name,
						   'filename':filename,
						   'H':H,
						   'fixed_sig':fixed_sig,
						   'skill_seq_len':skill_seq_len,
						  })

	ep_rewards = []
	random_goals = False
	tasks = ['bottom burner','top burner','light switch','slide cabinet','hinge cabinet','microwave','kettle']

	idx = [[x for x in range(num_embeddings)]]*skill_seq_len
	iterlist = itertools.product(*idx)
	idx_list = []
	for i in iterlist:
		idx_list.append(i)
	idx_list = np.array(idx_list)

	for j in range(1000):
		H=10
		if(random_goals):
			ep_tasks = sample(tasks,4)
			franka_reward_fn = FrankaRewardFn(ep_tasks)
			print('TASKS: ',ep_tasks)
		initial_state = env.reset()
		prev_states = np.expand_dims(initial_state,0)  # add dummy time dimension
		frames = []
		rewards = []
		t_since_last_save = 0
		state = initial_state
		for i in range(max_replans):
			print('==============================================')
			print('i: ', i)
			if(random_goals):
				cost_fn = lambda skill_idx: franka_plan_cost_fn(prev_states,skill_idx,skill_model,ep_tasks=ep_tasks)
			else:
				cost_fn = lambda skill_idx: franka_plan_cost_fn(prev_states,skill_idx,skill_model,prior_search=True)
	
			#skill_seq = exhaustive_search(skill_model, cost_fn, idx_list, batch_size)
			skill_seq = prior_search(skill_model, cost_fn, batch_size, skill_seq_len, n_iters=1)

			z = skill_seq[:1,:]
			skill_seq = skill_seq.unsqueeze(0)
			
			state,states_actual,actions,skill_rewards = run_skill_seq(skill_seq,env,state,skill_model)
			
			if(random_goals):
				skill_rewards = 0.0
				for h in range(H):
					r,_ = franka_reward_fn.step(states_actual[h])
					skill_rewards += r

			prev_states = np.concatenate([prev_states,states_actual[1:,:]],axis=0)
			print('states_actual.shape: ', states_actual.shape)
			rewards.append(skill_rewards)
			print('np.sum(rewards): ', np.sum(rewards))


		experiment.log_metric('reward',np.sum(rewards),step=j)
		ep_rewards.append(np.sum(rewards))
		print('EPISODE: ',j)
		print('MEAN REWARD: ',sum(ep_rewards)/(j+1))
		print('STD DEV: ',np.std(ep_rewards,ddof=1))
		experiment.log_metric('mean_reward',np.mean(ep_rewards),step=j)
