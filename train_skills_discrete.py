from comet_ml import Experiment
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.distributions.normal as Normal
from skill_model_vq import SkillModelVectorQuantized, SkillModelDiscrete
from utils import chunks
import config
import os

def train(model,optimizer):
	losses = []
	for batch_id, data in enumerate(train_loader):
		data = data.cuda()
		states = data[:,:,:model.state_dim]
		actions = data[:,:,model.state_dim:]	 # rest are actions

		loss = model.get_loss(states,actions)
		model.zero_grad()
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
			states = data[:,:,:model.state_dim]  # first state_dim elements are the state
			actions = data[:,:,model.state_dim:]	 # rest are actions

			embedding_loss, a_loss, sT_loss, total_loss, prior_loss  = model.get_losses(states, actions)

			# log losses
			losses.append(total_loss.item())
			s_T_losses.append(sT_loss.item())
			a_losses.append(a_loss.item())
			embedding_losses.append(embedding_loss.item())
			prior_losses.append(prior_loss.item())

	return np.mean(losses), np.mean(s_T_losses), np.mean(a_losses), np.mean(embedding_losses), np.mean(prior_losses)


batch_size = 100

h_dim = 256
z_dim = 4
num_embeddings = 4
lr = 5e-5
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
per_element_sigma = False

# env_name = 'antmaze-large-diverse-v0'
#env_name = 'kitchen-mixed-v0'
#env_name = 'maze2d-large-v1'
#env_name = 'carla-nocrash'
env_name = 'carla-nocrash2'

states = np.load('data/'+env_name+'/observations.npy')
next_states = np.load('data/'+env_name+'/next_observations.npy')
actions = np.load('data/'+env_name+'/actions.npy')
if env_name=='carla-nocrash' or env_name=='carla-nocrash2':
	terminals = np.load('data/'+env_name+'/terminals.npy')
terminals_train = None
terminals_test = None
N = states.shape[0]
state_dim = states.shape[1]
a_dim = actions.shape[1]
N_train = int((1-test_split)*N)
N_test = N - N_train
states_train  = states[:N_train,:]
next_states_train = next_states[:N_train,:]
actions_train = actions[:N_train,:]
if env_name=='carla-nocrash' or env_name=='carla-nocrash2':
	terminals_train =  terminals[:N_train,:]
states_test  = states[N_train:,:]
next_states_test = next_states[N_train:,:]
actions_test = actions[N_train:,:]
if env_name=='carla-nocrash' or env_name=='carla-nocrash2':
	terminals_test =  terminals[N_train:,:]
obs_chunks_train, action_chunks_train = chunks(states_train, next_states_train, actions_train, H, stride, terminals_train)
print('states_test.shape: ',states_test.shape)
print('MAKIN TEST SET!!!')
obs_chunks_test,  action_chunks_test  = chunks(states_test,  next_states_test,  actions_test,  H, stride, terminals_test)

experiment = Experiment(api_key = 'LVi0h2WLrDaeIC6ZVITGAvzyl', project_name = 'vq_skills')
experiment.add_tag('nocrash')

# First, instantiate a skill model

model = SkillModelDiscrete(state_dim, a_dim, z_dim, h_dim, num_embeddings, a_dist=a_dist,state_dec_stop_grad=False,beta=beta,alpha=alpha,max_sig=None,fixed_sig=None,ent_pen=0,encoder_type='state_action_sequence',state_decoder_type=state_decoder_type,init_state_dependent=init_state_dependent,per_element_sigma=per_element_sigma).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

filename = 'discrete_model_'+env_name+'_init_state_dep_'+str(init_state_dependent)+'_zdim_'+str(z_dim)+'_H_'+str(H)+'_l2reg_'+str(wd)+'_a_'+str(alpha)+'_b_'+str(beta)+'_per_el_sig_'+str(per_element_sigma)+'_log'

if load_from_checkpoint:
	PATH = os.path.join(config.ckpt_dir,filename+'_best_sT.pth')
	checkpoint = torch.load(PATH)
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer'])

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


inputs_train = torch.cat([obs_chunks_train, action_chunks_train],dim=-1)
inputs_test  = torch.cat([obs_chunks_test,  action_chunks_test], dim=-1)

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
min_test_a_loss = 10**10
for i in range(n_epochs):

	test_loss, test_s_T_loss, test_a_loss, test_embedding_loss, test_prior_loss = test(model)
	
	print("--------TEST---------")
	
	print('test_loss: ', test_loss)
	print('test_s_T_loss: ', test_s_T_loss)
	print('test_a_loss: ', test_a_loss)
	print('test_embedding_loss: ', test_embedding_loss)
	print('test_prior_loss: ', test_prior_loss)

	print(i)
	experiment.log_metric("test_loss", test_loss, step=i)
	experiment.log_metric("test_s_T_loss", test_s_T_loss, step=i)
	experiment.log_metric("test_a_loss", test_a_loss, step=i)
	experiment.log_metric("test_embedding_loss", test_embedding_loss, step=i)
	experiment.log_metric("test_prior_loss", test_prior_loss, step=i)
	
	if test_loss < min_test_loss:
		min_test_loss = test_loss
		checkpoint_path = os.path.join(config.ckpt_dir,filename+'_best.pth')
		# checkpoint_path = 'checkpoints/'+ filename + '_best.pth'
		torch.save({'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)

	if test_s_T_loss < min_test_s_T_loss:
		min_test_s_T_loss = test_s_T_loss
		checkpoint_path = os.path.join(config.ckpt_dir,filename+'_best_sT.pth')
		# checkpoint_path = 'checkpoints/'+ filename + '_best.pth'
		torch.save({'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)

	if test_a_loss < min_test_a_loss:
		min_test_a_loss = test_a_loss
		checkpoint_path = os.path.join(config.ckpt_dir,filename+'_best_a.pth')
		# checkpoint_path = 'checkpoints/'+ filename + '_best.pth'
		torch.save({'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)
	

	loss = train(model,optimizer)
	
	print("--------TRAIN---------")
	
	print('loss', loss)
	print(i)
	experiment.log_metric("train_loss", loss, step=i)

	if i % 10 == 0:
		
		checkpoint_path = os.path.join(config.ckpt_dir,filename+'.pth')
		# checkpoint_path = 'checkpoints/'+ filename + '.pth'
		torch.save({
							'model_state_dict': model.state_dict(),
							'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)

	