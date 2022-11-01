import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.distributions.normal as Normal
import random


def reparameterize(mean, std):
	eps = torch.normal(torch.zeros(mean.size()).cuda(), torch.ones(mean.size()).cuda())
	return mean + std*eps



def chunks(obs,next_obs,actions,H,stride):
	'''
	obs is a N x 4 array
	goals is a N x 2 array
	H is length of chunck
	stride is how far we move between chunks.  So if stride=H, chunks are non-overlapping.  If stride < H, they overlap
	'''
	
	obs_chunks = []
	action_chunks = []
	N = obs.shape[0]
	for i in range(N//stride - H):
		start_ind = i*stride
		end_ind = start_ind + H
		# If end_ind = 4000000, it goes out of bounds
		# this way start_ind is from 0-3999980 and end_ind is from 20-3999999
		# if end_ind == N:
		# 	end_ind = N-1
		
		obs_chunk = torch.tensor(obs[start_ind:end_ind,:],dtype=torch.float32)

		action_chunk = torch.tensor(actions[start_ind:end_ind,:],dtype=torch.float32)
		
		loc_deltas = obs_chunk[1:,:] - obs_chunk[:-1,:] #Franka or Maze2d
		
		norms = np.linalg.norm(loc_deltas,axis=-1)
		#USE VALUE FOR THRESHOLD CONDITION BASED ON ENVIRONMENT
		if np.all(norms <= 0.7): #Antmaze large 0.8 medium 0.67 / Franka partial 0.22 / Maze2d 0.6
			obs_chunks.append(obs_chunk)
			action_chunks.append(action_chunk)
		else:
			pass
			# print('NOT INCLUDING ',i)

	print('len(obs_chunks): ',len(obs_chunks))
	print('len(action_chunks): ',len(action_chunks))
			
	
	return torch.stack(obs_chunks),torch.stack(action_chunks)