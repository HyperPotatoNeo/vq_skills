import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def exhaustive_search(skill_model, cost_fn, idx_list, batch_size):
    with torch.no_grad():
        min_cost = 1000000
        skill_seq = torch.zeros((idx_list.shape[1],skill_model.vector_quantizer.z_dim)).cuda()
        for i in range((idx_list.shape[0]//batch_size)+1):
            if(i==(idx_list.shape[0]//batch_size)):
                batch_idx = idx_list[i*batch_size:]
            else:
                batch_idx = idx_list[i*batch_size:(i+1)*batch_size]
            batch_costs = cost_fn(batch_idx)
            min_id = torch.argmin(batch_costs)
            if(batch_costs[min_id].cpu().item() < min_cost):
                min_cost = batch_costs[min_id].cpu().item()
                min_idx = idx_list[min_id]
                for j in range(idx_list.shape[1]):
                    skill_seq[j,:] = skill_model.vector_quantizer.embedding.weight[min_idx[j]]
            #print(min_cost)
    return skill_seq