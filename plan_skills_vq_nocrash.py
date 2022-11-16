'''File where we will sample a set of waypionts, and plan a sequence of skills to have our pointmass travel through those waypoints'''

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
from math import pi
from search_skills import exhaustive_search, prior_search, max_prior
import itertools
from statsmodels.stats.proportion import proportion_confint
from environment.env import CarlaEnv
from environment.config.config import DefaultMainConfig
import cv2

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

device = torch.device('cuda:0')
config = DefaultMainConfig()
config.populate_config(
    observation_config = "VehicleDynamicsObstacleNoCameraConfig",
    action_config = "MergedSpeedScaledTanhConfig",
    reward_config = "Simple2RewardConfig",
    scenario_config = "NoCrashDenseTown01Config",
    testing = False,
    carla_gpu = 0,
    render_server=True
)

env = CarlaEnv(config = config)
data = {}
episodes = 100

dataset = {}
dataset['observations'] = np.load('data/carla-nocrash/observations.npy')
dataset['next_observations'] = np.load('data/carla-nocrash/observations.npy')
dataset['actions'] = np.load('data/carla-nocrash/actions.npy')
dataset['terminals'] = np.load('data/carla-nocrash/terminals.npy')
states = dataset['observations']
next_states = dataset['next_observations']
actions = dataset['actions']
terminals = dataset['terminals']

skill_seq_len = 1#6
H = 5#30#40
state_dim = dataset['observations'].shape[1]
a_dim = dataset['actions'].shape[1]
h_dim = 256
z_dim = 4
num_embeddings = 8
batch_size = 10
lr = 1e-4
wd = 0.0
state_dependent_prior = True
state_dec_stop_grad = True
beta = 1.0
alpha = 1.0
ent_pen = 0
max_sig = None
fixed_sig = 0.0#None #(Try None for antmaze)
a_dist = 'autoregressive'#'normal'
per_element_sigma = True
max_replans = 2000 // H
encoder_type = 'state_action_sequence'
state_decoder_type = 'mlp'
term_state_dependent_prior = False
init_state_dependent = True

filename = 'best_priorVQ_model_carla-nocrash_num_embeddings_8_init_state_dep_True_zdim_4_H_5_l2reg_0.0_a_1.0_b_1.0_per_el_sig_True_log_best.pth'

PATH = 'carla_checkpoints/'+filename

skill_model = SkillModelVectorQuantized(state_dim, a_dim, z_dim, h_dim, num_embeddings, a_dist=a_dist,state_dec_stop_grad=False,beta=beta,alpha=alpha,max_sig=None,fixed_sig=fixed_sig,ent_pen=0,encoder_type='state_action_sequence',state_decoder_type=state_decoder_type,init_state_dependent=init_state_dependent,per_element_sigma=per_element_sigma).cuda()

checkpoint = torch.load(PATH)
skill_model.load_state_dict(checkpoint['model_state_dict'])

experiment = Experiment(api_key = 'LVi0h2WLrDaeIC6ZVITGAvzyl', project_name = 'vq_skills')

s_init = env.reset()
s0_torch = torch.cat([torch.tensor(s_init,dtype=torch.float32).cuda().reshape(1,1,-1) for _ in range(batch_size)])

skill_seq = torch.zeros((1,skill_seq_len,z_dim),device=device)
print('skill_seq.shape: ', skill_seq.shape)
skill_seq.requires_grad = True

idx = [[x for x in range(num_embeddings)]]*skill_seq_len
iterlist = itertools.product(*idx)
idx_list = []
for i in iterlist:
    idx_list.append(i)
idx_list = np.array(idx_list)

def run_skill_seq(skill_seq,env,s0,model,capture_video=False,video_obj=None):
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
            action = model.decoder.ll_policy.numpy_policy(state,z)
            state,reward,done,info = env.step(action)
            if(capture_video):
                img = info["sensor.camera.rgb/top"]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                video_obj.write(img)
            skill_seq_states.append(state)
            if(done):
                return state, done
        states.append(skill_seq_states)

    states = np.stack(states)

    return state,done
        


execute_n_skills = 1
capture_video = False
for j in range(1000):
    video=cv2.VideoWriter('videos/town01/'+str(j)+'.mp4',fourcc,10,(512,512))
    print('EPISODE: ',j+1)
    state = env.reset(index = j)
    done = False
    while(not done):
        s_torch = torch.cat(batch_size*[torch.tensor(state,dtype=torch.float32,device=device).reshape((1,1,-1))])
        
        cost_fn = lambda skill_idx: skill_model.get_expected_cost_vq(s_torch, skill_idx, use_reward_model=True)
        skill_seq = exhaustive_search(skill_model, cost_fn, idx_list, batch_size)
        #cost_fn = lambda batch_size: skill_model.get_expected_cost_vq_prior(s_torch, batch_size, use_reward_model=True)
        #skill_seq = prior_search(skill_model, cost_fn, batch_size, skill_seq_len, n_iters=1)
        #skill_seq = max_prior(skill_model, s_torch)

        skill_seq = skill_seq[:execute_n_skills,:]
        skill_seq = skill_seq.unsqueeze(0)  
        state,done = run_skill_seq(skill_seq,env,state,skill_model,capture_video,video)
    video.release()