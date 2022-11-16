from comet_ml import Experiment
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.distributions.normal as Normal
from skill_model_vq import SkillModelVectorQuantized, AbstractReward
from utils import chunks, reward_chunks
import config
import os

def unison_shuffled_copies(states, z_q, rewards):
    p = np.random.permutation(states.shape[0])
    return states[p], z_q[p], rewards[p]

def weight_reset(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()

def train(model,optimizer):
	losses = []
	for batch_id, data in enumerate(train_loader):
		data = data.cuda()
		states = data[:,:,:model.state_dim]
		z_q = data[:,:,model.state_dim:model.state_dim+model.decoder.z_dim]	 # rest are actions
		rT_true = data[:,:,model.state_dim+model.decoder.z_dim:]

		loss = model.reward_model.get_loss(states,z_q,rT_true)
		model.reward_model.zero_grad()
		loss.backward()
		optimizer.step()
		# log losses
		losses.append(loss.item())
		
	return np.mean(losses)

def test(model):
	losses = []
	s_T_losses = []
	a_losses = []
	embedding_losses = []
	prior_losses = []

	with torch.no_grad():
		for batch_id, data in enumerate(test_loader):
			data = data.cuda()
			states = data[:,:,:model.state_dim]
			z_q = data[:,:,model.state_dim:model.state_dim+model.decoder.z_dim]	 # rest are actions
			rT_true = data[:,:,model.state_dim+model.decoder.z_dim:]

			loss  = model.reward_model.get_loss(states,z_q,rT_true)

			# log losses
			losses.append(loss.item())

	return np.mean(losses)


batch_size = 100

h_dim = 256
z_dim = 4
num_embeddings = 8
lr = 1e-3
wd = 0.0
beta = 1.0
alpha = 1.0
H = 5
stride = 1
n_epochs = 50000
test_split = .2
a_dist = 'autoregressive'
state_dependent_prior = True
encoder_type = 'state_action_sequence'
state_decoder_type = 'mlp'
init_state_dependent = True
load_from_checkpoint = False
per_element_sigma = True

#env_name = 'antmaze-large-diverse-v0'
#env_name = 'kitchen-mixed-v0'
#env_name = 'maze2d-large-v1'
env_name = 'carla-nocrash'

states = np.load('dagger_data/observations.npy')
rewards = np.load('dagger_data/rewards.npy')
#rewards = (rewards-np.mean(rewards))/np.std(rewards)
z_q = np.load('dagger_data/skills.npy')

states, z_q, rewards = unison_shuffled_copies(states, z_q, rewards)

terminals_train = None
terminals_test = None
N = states.shape[0]
state_dim = states.shape[1]
a_dim = 2
N_train = int((1-test_split)*N)
N_test = N - N_train
states_train  = states[:N_train,:]
rewards_train = rewards[:N_train,:]
z_q_train = z_q[:N_train,:]

states_test  = states[N_train:,:]
rewards_test = rewards[N_train:,:]
z_q_test = z_q[N_train:,:]

obs_chunks_train, z_q_chunks_train, rT_true_train = reward_chunks(states_train, rewards_train, z_q_train, H, stride)
print(rT_true_train.shape)
print('states_test.shape: ',states_test.shape)
print('MAKIN TEST SET!!!')
obs_chunks_test, z_q_chunks_test, rT_true_test  = reward_chunks(states_test,  rewards_test,  z_q_test,  H, stride)

experiment = Experiment(api_key = 'LVi0h2WLrDaeIC6ZVITGAvzyl', project_name = 'vq_skills')
experiment.add_tag('reward_model z4n8')

# First, instantiate a skill model

filename = 'best_priorVQ_model_carla-nocrash_num_embeddings_8_init_state_dep_True_zdim_4_H_5_l2reg_0.0_a_1.0_b_1.0_per_el_sig_True_log_best.pth'

model = SkillModelVectorQuantized(state_dim, a_dim, z_dim, h_dim, num_embeddings, a_dist=a_dist,state_dec_stop_grad=False,beta=beta,alpha=alpha,max_sig=None,fixed_sig=None,ent_pen=0,encoder_type='state_action_sequence',state_decoder_type=state_decoder_type,init_state_dependent=init_state_dependent,per_element_sigma=per_element_sigma).cuda()
PATH = os.path.join(config.carla_ckpt_dir,filename)
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
#model.reward_model = AbstractReward(state_dim,z_dim,h_dim).cuda()
model.reward_model.apply(weight_reset)

optimizer = torch.optim.Adam(model.reward_model.parameters(), lr=lr, weight_decay=wd)

experiment.log_parameters({'lr':lr,
							'h_dim':h_dim,
							'state_dependent_prior':state_dependent_prior,
							'z_dim':z_dim,
							'H':H,
							'a_dim':a_dim,
							'state_dim':state_dim,
							'l2_reg':wd,
							'beta':beta,
							'alpha':alpha,
							'env_name':env_name,
							'filename':filename,
							'encoder_type':encoder_type,
							'state_decoder_type':state_decoder_type,
							'per_element_sigma':per_element_sigma})


inputs_train = torch.cat([obs_chunks_train, z_q_chunks_train, rT_true_train],dim=-1)
inputs_test  = torch.cat([obs_chunks_test, z_q_chunks_test, rT_true_test], dim=-1)

train_data = TensorDataset(inputs_train)
test_data  = TensorDataset(inputs_test)

train_loader = DataLoader(
	inputs_train,
	batch_size=batch_size,
	num_workers=0)

test_loader = DataLoader(
	inputs_test,
	batch_size=batch_size,
	num_workers=0)

min_test_loss = 10**10
min_test_s_T_loss = 10**10
min_test_prior_loss = 10**10
for i in range(n_epochs):

	test_loss = test(model)

	print("--------TEST---------")
	
	print('test_loss: ', test_loss)

	print(i)
	experiment.log_metric("test_loss", test_loss, step=i)
	
	if test_loss < min_test_loss:
		min_test_loss = test_loss
		checkpoint_path = os.path.join(config.carla_ckpt_dir,filename)
		# checkpoint_path = 'checkpoints/'+ filename + '_best.pth'
		torch.save({'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)


	loss = train(model,optimizer)
	
	print("--------TRAIN---------")
	
	print('loss', loss)
	print(i)
	experiment.log_metric("train_loss", loss, step=i)
