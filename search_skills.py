import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.categorical as Categorical

def exhaustive_search(skill_model, cost_fn, idx_list, batch_size):
    with torch.no_grad():
        min_cost = 1000000
        skill_seq = torch.zeros((idx_list.shape[1],skill_model.vector_quantizer.z_dim)).cuda()
        for i in range((idx_list.shape[0]//batch_size+1)):#+1
            if(i==(idx_list.shape[0]//batch_size)):
                batch_idx = idx_list[i*batch_size:]
            else:
                batch_idx = idx_list[i*batch_size:(i+1)*batch_size]
            batch_costs = cost_fn(batch_idx)
            #print(batch_costs)
            min_id = torch.argmin(batch_costs)
            #print(min_id)
            if(batch_costs[min_id].cpu().item() < min_cost):
                min_cost = batch_costs[min_id].cpu().item()
                min_idx = idx_list[i*batch_size+min_id]
                for j in range(idx_list.shape[1]):
                    skill_seq[j,:] = skill_model.vector_quantizer.embedding.weight[min_idx[j]]
                    #random = np.random.binomial(1,0.2)
                    #if(random==1):
                    #    skill_seq[j,:] = skill_model.vector_quantizer.embedding.weight[5]
        #print(min_cost)
        #print(min_id)
    return skill_seq

def prior_search(skill_model, cost_fn, batch_size, skill_seq_len, n_iters=1):
    with torch.no_grad():
        min_cost = 1000000
        skill_seq = torch.zeros((skill_seq_len,skill_model.vector_quantizer.z_dim)).cuda()
        skill_idx= np.zeros((batch_size,skill_seq_len))
        for i in range(n_iters):
            idx_list, batch_costs = cost_fn(skill_idx)
            min_id = torch.argmin(batch_costs)
            if(batch_costs[min_id].cpu().item() < min_cost):
                min_idx = idx_list[min_id.cpu().item()]
                min_cost = batch_costs[min_id].cpu().item()
                for j in range(skill_seq_len):
                    skill_seq[j,:] = skill_model.vector_quantizer.embedding.weight[min_idx[j].astype(int)]
    print(min_cost)
    return skill_seq

def max_prior(skill_model,s0,i=0):
    with torch.no_grad():
        _,dist = skill_model.prior(s0)
        cat_dist = Categorical.Categorical(torch.squeeze(dist,dim=1))
        idx = cat_dist.sample().cpu().numpy()
        #if(i>30):
        #    idx=1
        #else:
        #    idx=0
        #idx = np.random.randint(0,4)
        #idx = torch.argmax(dist[0,0])
        #print(idx)
        skill_seq = torch.zeros((1,skill_model.vector_quantizer.z_dim)).cuda()
        skill_seq[0,:] = skill_model.vector_quantizer.embedding.weight[idx]
    return skill_seq