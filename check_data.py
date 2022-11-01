import numpy as np

#env_name = 'antmaze-large-diverse-v0'
env_name = 'kitchen-partial-v0'
#env_name = 'kitchen-mixed-v0'
#env_name = 'maze2d-large-v1'

obs = np.load('data/'+env_name+'/observations.npy')
next_obs = np.load('data/'+env_name+'/next_observations.npy')

transition_norm = np.linalg.norm(obs[:,:]-next_obs[:,:], axis=1)

sorted_norm = np.sort(transition_norm)

for i in range(sorted_norm.shape[0]):
	print(sorted_norm[i])
