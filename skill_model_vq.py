import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
from torch.distributions.transformed_distribution import TransformedDistribution
import torch.distributions.normal as Normal
import torch.distributions.categorical as Categorical
import torch.distributions.mixture_same_family as MixtureSameFamily
import torch.distributions.kl as KL
import matplotlib.pyplot as plt
from utils import reparameterize

class AbstractDynamics(nn.Module):
    '''
    P(s_T|s_0,z) is our "abstract dynamics model", because it predicts the resulting state transition over T timesteps given a skill 
    (so similar to regular dynamics model, but in skill space and also temporally extended)
    See Encoder and Decoder for more description
    '''
    def __init__(self,state_dim,z_dim,h_dim,init_state_dependent=True,per_element_sigma=True):

        super(AbstractDynamics,self).__init__()
        
        self.init_state_dependent = init_state_dependent
        if init_state_dependent:
            self.layers = nn.Sequential(nn.Linear(state_dim+z_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,h_dim),nn.ReLU())
        else:
            self.layers = nn.Sequential(nn.Linear(z_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,h_dim),nn.ReLU())
        #self.mean_layer = nn.Linear(h_dim,state_dim)
        self.mean_layer = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,state_dim))
        #self.sig_layer  = nn.Sequential(nn.Linear(h_dim,state_dim),nn.Softplus())
        if per_element_sigma:
            self.sig_layer  = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,state_dim),nn.Softplus())
        else:
            self.sig_layer = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,1),nn.Softplus())

        self.state_dim = state_dim
        self.per_element_sigma = per_element_sigma

    def forward(self,s0,z):

        '''
        INPUTS:
            s0: batch_size x 1 x state_dim initial state (first state in execution of skill)
            z:  batch_size x 1 x z_dim "skill"/z
        OUTPUTS: 
            sT_mean: batch_size x 1 x state_dim tensor of terminal (time=T) state means
            sT_sig:  batch_size x 1 x state_dim tensor of terminal (time=T) state standard devs
        '''

        if self.init_state_dependent:
            # concatenate s0 and z
            s0_z = torch.cat([s0,z],dim=-1)
            # pass s0_z through layers
            feats = self.layers(s0_z)
        else:
            feats = self.layers(z)
        # get mean and stand dev of action distribution
        sT_mean = self.mean_layer(feats)
        sT_sig  = self.sig_layer(feats)

        if not self.per_element_sigma:
            # sT_sig has shape batch_size x 1 x 1
            # tile sT_sig along final dimension, return it
            sT_sig = torch.cat(self.state_dim*[sT_sig],dim=-1)

        return sT_mean,sT_sig

class LowLevelPolicy(nn.Module):
    '''
    P(a_t|s_t,z) is our "low-level policy", so this is the feedback policy the agent runs while executing skill described by z.
    See Encoder and Decoder for more description
    '''
    def __init__(self,state_dim,a_dim,z_dim,h_dim,a_dist,max_sig=None,fixed_sig=None):

        super(LowLevelPolicy,self).__init__()
        
        self.layers = nn.Sequential(nn.Linear(state_dim+z_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,h_dim),nn.ReLU())
        #self.mean_layer = nn.Linear(h_dim,a_dim)
        self.mean_layer = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,a_dim))
        #self.sig_layer  = nn.Sequential(nn.Linear(h_dim,a_dim),nn.Softplus())
        self.sig_layer  = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,a_dim))
        self.a_dist = a_dist
        self.a_dim = a_dim
        self.max_sig = max_sig
        self.fixed_sig = fixed_sig



    def forward(self,state,z):
        '''
        INPUTS:
            state: batch_size x T x state_dim tensor of states 
            z:     batch_size x 1 x z_dim tensor of states
        OUTPUTS:
            a_mean: batch_size x T x a_dim tensor of action means for each t in {0.,,,.T}
            a_sig:  batch_size x T x a_dim tensor of action standard devs for each t in {0.,,,.T}
        '''
        # tile z along time axis so dimension matches state
        z_tiled = z.tile([1,state.shape[-2],1]) #not sure about this 

        # Concat state and z_tiled
        state_z = torch.cat([state,z_tiled],dim=-1)
        # pass z and state through layers
        feats = self.layers(state_z)
        # get mean and stand dev of action distribution
        a_mean = self.mean_layer(feats)
        if self.max_sig is None:
            a_sig  = nn.Softplus()(self.sig_layer(feats))
        else:
            a_sig = self.max_sig * nn.Sigmoid()(self.sig_layer(feats))

        if self.fixed_sig is not None:
            a_sig = self.fixed_sig*torch.ones_like(a_sig)

        return a_mean, a_sig
    
    def numpy_policy(self,state,z):
        '''
        maps state as a numpy array and z as a pytorch tensor to a numpy action
        '''
        state = torch.reshape(torch.tensor(state,device=torch.device('cuda:0'),dtype=torch.float32),(1,1,-1))
        
        a_mean,a_sig = self.forward(state,z)
        action = self.reparameterize(a_mean,a_sig)
        if self.a_dist == 'tanh_normal':
            action = nn.Tanh()(action)
        action = action.detach().cpu().numpy()
        
        return action.reshape([self.a_dim,])
     
    def reparameterize(self, mean, std):
        eps = torch.normal(torch.zeros(mean.size()).cuda(), torch.ones(mean.size()).cuda())
        return mean + std*eps

class AutoregressiveLowLevelPolicy(nn.Module):
    '''
    P(a_t|s_t,z) is our "low-level policy", so this is the feedback policy the agent runs while executing skill described by z.
    See Encoder and Decoder for more description
    '''
    def __init__(self,state_dim,a_dim,z_dim,h_dim,max_sig=None,fixed_sig=None):

        super(AutoregressiveLowLevelPolicy,self).__init__()

        # we'll need a_dim different low-level policies, one for each action element
        self.policy_components = nn.ModuleList([LowLevelPolicy(state_dim+i,1,z_dim,h_dim,a_dist='normal',max_sig=max_sig,fixed_sig=fixed_sig) for i in range(a_dim)])

        self.a_dim = a_dim

        self.a_dist = 'autoregressive'

        print('!!!!!!!!!!!! CREATING AUTOREGRESSIVE LL POLICY!!!!!!!!!!!!!!!!!!!')
        


    def forward(self,state,actions,z):
        '''
        INPUTS:
            state: batch_size x T x state_dim tensor of states 
            action: batch_size x T x a_dim tensor of actions
            z:     batch_size x 1 x z_dim tensor of states
        OUTPUTS:
            a_mean: batch_size x T x a_dim tensor of action means for each t in {0.,,,.T}
            a_sig:  batch_size x T x a_dim tensor of action standard devs for each t in {0.,,,.T}
        
        Iterate through each low level policy component.
        The ith element gets to condition on all elements up to but NOT including a_i
        '''
        # tile z along time axis so dimension matches state
        # z_tiled = z.tile([1,state.shape[-2],1]) #not sure about this 
        a_means = []
        a_sigs = []
        for i in range(self.a_dim):
            # Concat state, and a up to i.  state_a takes place of state in orginary policy.
            state_a = torch.cat([state,actions[:,:,:i]],dim=-1)
            # pass through ith policy component
            a_mean_i,a_sig_i = self.policy_components[i](state_a,z)  # these are batch_size x T x 1
            # add to growing list of policy elements
            a_means.append(a_mean_i)
            a_sigs.append(a_sig_i)

        a_means = torch.cat(a_means,dim=-1)
        a_sigs  = torch.cat(a_sigs, dim=-1)
        return a_means, a_sigs
    
    def sample(self,state,z):
        # tile z along time axis so dimension matches state
        # z_tiled = z.tile([1,state.shape[-2],1]) #not sure about this 
        actions = []
        for i in range(self.a_dim):
            # Concat state, a up to i, and z_tiled
            state_a = torch.cat([state]+actions,dim=-1)
            # pass through ith policy component
            a_mean_i,a_sig_i = self.policy_components[i](state_a,z)  # these are batch_size x T x 1
            a_i = reparameterize(a_mean_i,a_sig_i)
            actions.append(a_i)

        return torch.cat(actions,dim=-1)

    
    def numpy_policy(self,state,z):
        '''
        maps state as a numpy array and z as a pytorch tensor to a numpy action
        '''
        state = torch.reshape(torch.tensor(state,device=torch.device('cuda:0'),dtype=torch.float32),(1,1,-1))
        
        action = self.sample(state,z)
        action = action.detach().cpu().numpy()
        
        return action.reshape([self.a_dim,])


class Encoder(nn.Module):
    '''
    Encoder module.
    We can try the following architecture initially:
    -Concat states+actions
    -Pass through linear embedding
    -Pass through bidirectional RNN
    -Pass output of bidirectional RNN through 2 linear layers, one to get mean of z and one to get stand dev (we're estimating one z ("skill") for entire episode)
    '''
    def __init__(self,state_dim,a_dim,z_dim,h_dim,n_gru_layers=4):
        super(Encoder, self).__init__()


        self.state_dim = state_dim # state dimension
        self.a_dim = a_dim # action dimension

        self.emb_layer  = nn.Sequential(nn.Linear(state_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,h_dim),nn.ReLU())
        self.rnn        = nn.GRU(h_dim+a_dim,h_dim,batch_first=True,bidirectional=True,num_layers=n_gru_layers)
        #self.mean_layer = nn.Linear(h_dim,z_dim)
        self.mean_layer = nn.Sequential(nn.Linear(2*h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,z_dim))

    def forward(self,states,actions):

        '''
        Takes a sequence of states and actions, and infers the distribution over latent skill variable, z
        
        INPUTS:
            states: batch_size x T x state_dim state sequence tensor
            actions: batch_size x T x a_dim action sequence tensor
        OUTPUTS:
            z_mean: batch_size x 1 x z_dim tensor indicating mean of z distribution
            z_sig:  batch_size x 1 x z_dim tensor indicating standard deviation of z distribution
        '''
        
        s_emb = self.emb_layer(states)
        # through rnn
        s_emb_a = torch.cat([s_emb,actions],dim=-1)
        feats,_ = self.rnn(s_emb_a)
        hn = feats[:,-1:,:]
        z_mean = self.mean_layer(hn)

        return z_mean

class Decoder(nn.Module):
    '''
    Decoder module.
    Decoder takes states, actions, and a sampled z and outputs parameters of P(s_T|s_0,z) and P(a_t|s_t,z) for all t in {0,...,T}
    P(s_T|s_0,z) is our "abstract dynamics model", because it predicts the resulting state transition over T timesteps given a skill 
    (so similar to regular dynamics model, but in skill space and also temporally extended)
    P(a_t|s_t,z) is our "low-level policy", so this is the feedback policy the agent runs while executing skill described by z.
    We can try the following architecture:
    -embed z
    -Pass into fully connected network to get "state T features"
    '''
    def __init__(self,state_dim,a_dim,z_dim,h_dim, a_dist,state_dec_stop_grad, max_sig, fixed_sig, state_decoder_type, init_state_dependent, per_element_sigma):

        super(Decoder,self).__init__()
        
        print('in decoder a_dist: ', a_dist)
        self.state_dim = state_dim
        self.a_dim = a_dim
        self.z_dim = z_dim

        if state_decoder_type == 'mlp':
            self.abstract_dynamics = AbstractDynamics(state_dim,z_dim,h_dim,init_state_dependent=init_state_dependent,per_element_sigma=per_element_sigma)
        elif state_decoder_type == 'autoregressive':
            self.abstract_dynamics = AutoregressiveStateDecoder(state_dim,z_dim,h_dim)
        else:
            print('PICK VALID STATE DECODER TYPE!!!')
            assert False
        if a_dist != 'autoregressive':
            self.ll_policy = LowLevelPolicy(state_dim,a_dim,z_dim,h_dim, a_dist, max_sig = max_sig, fixed_sig=fixed_sig)
        else:
            print('making autoregressive policy')
            self.ll_policy = AutoregressiveLowLevelPolicy(state_dim,a_dim,z_dim,h_dim,max_sig=None,fixed_sig=None)

        self.emb_layer  = nn.Linear(state_dim+z_dim,h_dim)
        self.fc = nn.Sequential(nn.Linear(state_dim+z_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,h_dim),nn.ReLU())

        self.state_dec_stop_grad = state_dec_stop_grad

        self.state_decoder_type = state_decoder_type
        self.a_dist = a_dist

        
    def forward(self,states,actions,z):

        '''
        INPUTS: 
            states: batch_size x T x state_dim state sequence tensor
            z:      batch_size x 1 x z_dim sampled z/skill variable
        OUTPUTS:
            sT_mean: batch_size x 1 x state_dim tensor of terminal (time=T) state means
            sT_sig:  batch_size x 1 x state_dim tensor of terminal (time=T) state standard devs
            a_mean: batch_size x T x a_dim tensor of action means for each t in {0.,,,.T}
            a_sig:  batch_size x T x a_dim tensor of action standard devs for each t in {0.,,,.T}
        '''
        
        s_0 = states[:,0:1,:]
        s_T = states[:,-1:,:]

        if self.a_dist != 'autoregressive':
            a_mean,a_sig = self.ll_policy(states,z)
        else:
            a_mean,a_sig = self.ll_policy(states,actions,z)

        if self.state_dec_stop_grad:
            z = z.detach()
        
        
        if self.state_decoder_type == 'autoregressive':
            sT_mean,sT_sig = self.abstract_dynamics(s_T.detach(),s_0.detach(),z.detach())
        elif self.state_decoder_type == 'mlp':
            sT_mean,sT_sig = self.abstract_dynamics(s_0.detach(),z.detach())
        else:
            print('PICK VALID STATE DECODER TYPE!!!')
            assert False
        
        return sT_mean,sT_sig,a_mean,a_sig


class GenerativeModel(nn.Module):

    def __init__(self,decoder,prior):
        super().__init__()
        self.decoder = decoder
        self.prior = prior

    def forward(self):
        pass


class Prior(nn.Module):
    '''
    Decoder module.
    Decoder takes states, actions, and a sampled z and outputs parameters of P(s_T|s_0,z) and P(a_t|s_t,z) for all t in {0,...,T}
    P(s_T|s_0,z) is our "abstract dynamics model", because it predicts the resulting state transition over T timesteps given a skill 
    (so similar to regular dynamics model, but in skill space and also temporally extended)
    P(a_t|s_t,z) is our "low-level policy", so this is the feedback policy the agent runs while executing skill described by z.
    We can try the following architecture:
    -embed z
    -Pass into fully connected network to get "state T features"
    '''
    def __init__(self,state_dim,z_dim,h_dim,goal_conditioned=False,goal_dim=2):

        super(Prior,self).__init__()
        
        self.state_dim = state_dim
        self.z_dim = z_dim
        self.goal_conditioned = goal_conditioned
        if(self.goal_conditioned):
            self.goal_dim = goal_dim
        else:
            self.goal_dim = 0
        self.layers = nn.Sequential(nn.Linear(state_dim+self.goal_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,h_dim),nn.ReLU())
        #self.mean_layer = nn.Linear(h_dim,z_dim)
        self.mean_layer = nn.Sequential(nn.Linear(h_dim,h_dim),nn.ReLU(),nn.Linear(h_dim,z_dim),nn.Softmax())
        
    def forward(self,s0,goal=None):

        '''
        INPUTS: 
            states: batch_size x T x state_dim state sequence tensor
            
        OUTPUTS:
            z_mean: batch_size x 1 x state_dim tensor of z means
            z_sig:  batch_size x 1 x state_dim tensor of z standard devs
            
        '''
        if(self.goal_conditioned):
            s0 = torch.cat([s0,goal],dim=-1)
        feats = self.layers(s0)
        # get mean and stand dev of action distribution
        z_mean = self.mean_layer(feats)
        z_sig  = self.sig_layer(feats)

        return z_mean, z_sig


class VectorQuantizer(nn.Module):
    def __init__(self,z_dim,num_embeddings,beta):
        super(VectorQuantizer,self).__init__()
        self.z_dim = z_dim
        self.embedding_dim = z_dim
        self.num_embeddings = num_embeddings
        self.beta = beta

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, z):
        z_flat = torch.squeeze(z,dim=1)
        d = torch.sum(z_flat ** 2, dim=1, keepdim=True)+torch.sum(self.embedding.weight**2, dim=1) - 2*torch.matmul(z_flat, self.embedding.weight.t())
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.num_embeddings).cuda()
        min_encodings.scatter_(1, min_encoding_indices, 1)

        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z_flat.shape)
        embedding_loss = torch.mean((z_q.detach()-z_flat)**2) + self.beta*torch.mean((z_q - z_flat.detach()) ** 2)
        z_q = z_flat + (z_q - z_flat).detach()
        z_q = z_q.view(z.shape)

        return z_q, min_encoding_indices, embedding_loss


class SkillModelVectorQuantized(nn.Module):
    def __init__(self,state_dim,a_dim,z_dim,h_dim,num_embeddings=128,a_dist='normal',state_dec_stop_grad=False,beta=0.25,alpha=1.0,max_sig=None,fixed_sig=None,ent_pen=0,encoder_type='state_action_sequence',state_decoder_type='mlp',init_state_dependent=True,per_element_sigma=True):
        super(SkillModelVectorQuantized, self).__init__()

        print('a_dist: ', a_dist)
        self.state_dim = state_dim # state dimension
        self.a_dim = a_dim # action dimension
        self.encoder_type = encoder_type
        self.state_dec_stop_grad = state_dec_stop_grad
        
        if encoder_type == 'state_action_sequence':
            self.encoder = Encoder(state_dim,a_dim,z_dim,h_dim)
        elif encoder_type == 's0sT':
            self.encoder = S0STEncoder(state_dim,a_dim,z_dim,h_dim)
        elif encoder_type == 'state_sequence':
            self.encoder = StateSeqEncoder(state_dim,a_dim,z_dim,h_dim)
        else:
            print('INVALID ENCODER TYPE!!!!')
            assert False

        self.decoder = Decoder(state_dim,a_dim,z_dim,h_dim, a_dist, state_dec_stop_grad,max_sig=max_sig,fixed_sig=fixed_sig,state_decoder_type=state_decoder_type,init_state_dependent=init_state_dependent,per_element_sigma=per_element_sigma)
        #self.prior   = Prior(state_dim,z_dim,h_dim)
        self.vector_quantizer = VectorQuantizer(z_dim,num_embeddings,beta)
        self.beta    = beta
        self.alpha   = alpha
        self.ent_pen = ent_pen

        if ent_pen != 0:
            assert not state_dec_stop_grad

    def forward(self,states,actions):
        
        '''
        Takes states and actions, returns the distributions necessary for computing the objective function
        INPUTS:
            states: batch_size x T x state_dim state sequence tensor
            actions: batch_size x T x a_dim action sequence tensor
        OUTPUTS:
            s_T_mean:     batch_size x 1 x state_dim tensor of means of "decoder" distribution over terminal states
            S_T_sig:      batch_size x 1 x state_dim tensor of standard devs of "decoder" distribution over terminal states
            a_means:      batch_size x T x a_dim tensor of means of "decoder" distribution over actions
            a_sigs:       batch_size x T x a_dim tensor of stand devs
            z_post_means: batch_size x 1 x z_dim tensor of means of z posterior distribution
            z_post_sigs:  batch_size x 1 x z_dim tensor of stand devs of z posterior distribution 
        '''

        # STEP 1. Encode states and actions to get posterior over z
        z_post_means,z_post_sigs = self.encoder(states,actions)
        # STEP 2. sample z from posterior 
        z_sampled = self.reparameterize(z_post_means,z_post_sigs)

        # STEP 3. Pass z_sampled and states through decoder 
        s_T_mean, s_T_sig, a_means, a_sigs = self.decoder(states,actions,z_sampled) # 5/4/22 add actions as argument here for autoregressive policy



        return s_T_mean, s_T_sig, a_means, a_sigs, z_post_means, z_post_sigs


    def get_loss(self,states,actions):

        batch_size,T,_ = states.shape
        denom = T*batch_size
        
        z_post_means = self.encoder(states,actions)

        z_q, min_encoding_indices, embedding_loss = self.vector_quantizer(z_post_means)

        sT_mean, sT_sig, a_means, a_sigs = self.decoder(states,actions,z_q)

        sT_dist  = Normal.Normal(sT_mean,sT_sig)
        a_dist    = Normal.Normal(a_means,a_sigs)

        sT = states[:,-1:,:]
        sT_loss = -torch.sum(sT_dist.log_prob(sT)) / denom
        a_loss =  -torch.sum(a_dist.log_prob(actions)) / denom

        return self.alpha*sT_loss + a_loss + embedding_loss

    
    def get_losses(self,states,actions):
        '''
        Computes various components of the loss:
        L = E_q [log P(s_T|s_0,z)] 
          + E_q [sum_t=0^T P(a_t|s_t,z)] 
          - D_kl(q(z|s_0,...,s_T,a_0,...,a_T)||P(z_0|s_0))
        Distributions we need:
        '''

        batch_size,T,_ = states.shape
        denom = T*batch_size
        
        z_post_means = self.encoder(states,actions)

        z_q, min_encoding_indices, embedding_loss = self.vector_quantizer(z_post_means)

        sT_mean, sT_sig, a_means, a_sigs = self.decoder(states,actions,z_q)

        sT_dist  = Normal.Normal(sT_mean,sT_sig)
        a_dist    = Normal.Normal(a_means,a_sigs)

        sT = states[:,-1:,:]
        sT_loss = -torch.sum(sT_dist.log_prob(sT)) / denom
        a_loss =  -torch.sum(a_dist.log_prob(actions)) / denom
        total_loss = self.alpha*sT_loss + a_loss + embedding_loss

        return embedding_loss, a_loss, sT_loss, total_loss
            

    def get_expected_cost_for_cem(self, s0, skill_seq, goal_state, use_epsilons=True, plot=False, length_cost=0, var_pen=0.0):
        '''
        s0 is initial state, batch_size x 1 x s_dim
        skill sequence is a batch_size x skill_seq_len x z_dim tensor that representents a skill_seq_len sequence of skills
        '''
        # tile s0 along batch dimension
        #s0_tiled = s0.tile([1,batch_size,1])
        batch_size = s0.shape[0]
        goal_state = torch.cat(batch_size * [goal_state],dim=0)
        s_i = s0
        
        skill_seq_len = skill_seq.shape[1]
        pred_states = [s_i]
        # costs = torch.zeros(batch_size,device=s0.device)
        costs = [torch.mean((s_i[:,:,:2] - goal_state[:,:,:2])**2,dim=-1).squeeze()]
        # costs = (lengths == 0)*torch.mean((s_i[:,:,:2] - goal_state[:,:,:2])**2,dim=-1).squeeze()
        var_cost = 0.0
        for i in range(skill_seq_len):
            
            # z_i = skill_seq[:,i:i+1,:] # might need to reshape
            if use_epsilons:
                mu_z, sigma_z = self.prior(s_i)
                
                z_i = mu_z + sigma_z*skill_seq[:,i:i+1,:]
            else:
                z_i = skill_seq[:,i:i+1,:]
            
            s_mean, s_sig = self.decoder.abstract_dynamics(s_i,z_i)

            var_cost += var_pen*var_cost 
            
            # sample s_i+1 using reparameterize
            s_sampled = s_mean
            # s_sampled = self.reparameterize(s_mean, s_sig)
            s_i = s_sampled

            cost_i = torch.mean((s_i[:,:,:2] - goal_state[:,:,:2])**2,dim=-1).squeeze() + (i+1)*length_cost
            costs.append(cost_i)
            
            pred_states.append(s_i)
        
        costs = torch.stack(costs,dim=1)  # should be a batch_size x T or batch_size x T 
        costs,_ = torch.min(costs,dim=1)  # should be of size batch_size

        return costs + var_cost

    
    def reparameterize(self, mean, std):
        eps = torch.normal(torch.zeros(mean.size()).cuda(), torch.ones(mean.size()).cuda())
        return mean + std*eps