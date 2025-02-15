import sys
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

"""

========================================================相似度计算函数============================================

"""


class cal_similarity(nn.Module):
    def __init__(self, args, temperature):
        super(cal_similarity, self).__init__()
        self.temperature = temperature
        self.args = args
        
        self.target_dim = args.target_dim
        self.entity_dim = args.target_dim * args.entity_mul
        self.relation_dim = args.target_dim * args.relation_mul
        self.layer_mul = 2
        
        self.query_distance = nn.Sequential(
            nn.Linear((self.entity_dim + self.relation_dim), (self.entity_dim + self.relation_dim) * self.layer_mul),
            nn.LeakyReLU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear((self.entity_dim + self.relation_dim) * self.layer_mul, 1),
        )
        
        self.tail_distance = nn.Sequential(
            nn.Linear((self.entity_dim + self.entity_dim), (self.entity_dim + self.entity_dim) * self.layer_mul),
            nn.LeakyReLU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear((self.entity_dim + self.entity_dim) * self.layer_mul, 1),
        )
        
    def cosine_similarity(self, ehr, et):
        if ehr.shape[1] < et.shape[1]: 
            ehr = ehr.expand(-1, et.shape[1], -1)
        else:
            et = et.expand(-1, ehr.shape[1], -1)
        sim = F.cosine_similarity(ehr, et, dim=-1)
        return sim
    
    def SCCF_similarity1(self, ehr, et):
        dot_product = torch.sum(ehr * et, dim=-1)
        norm_product = torch.norm(ehr, p=2, dim=-1) * torch.norm(et, p=2, dim=-1)
        sim = torch.exp(dot_product / (self.temperature * norm_product))
        return sim
    
    def SCCF_similarity3(self, ehr, et):
        dot_product = torch.sum(ehr * et, dim=-1)
        norm_product = torch.norm(ehr, p=2, dim=-1) * torch.norm(et, p=2, dim=-1)
        sim = torch.exp(dot_product / (self.temperature * norm_product)) + torch.exp((dot_product / norm_product) ** 3 / self.temperature)
        return sim
    
    def similarity1(self, ehr, et):
        return self.cosine_similarity(ehr, et)
    
    def similarity3(self, ehr, et):
        return self.cosine_similarity(ehr, et) + (self.cosine_similarity(ehr, et))**3
    
    def norm_distance(self, ehr, et, norm=2):
        # ehr.shape=[batch,1,dim] et.shape=[batch,nneg+1,dim]
        ehr_norm = torch.norm(ehr, p=norm, dim=-1)  
        et_norm = torch.norm(et, p=norm, dim=-1)
        distance = torch.abs(ehr_norm - et_norm)
        distance = -1 * distance
        return distance
    
    def norm_distance2(self, eh, er, ehr, et):
        query_distance = self.query_distance(torch.cat((eh,er),dim=-1))
        ehr = ehr.expand(-1, et.size(1), -1)
        tail_distance = self.tail_distance(torch.cat((ehr, et), dim=-1))
        distance = query_distance - tail_distance
        distance = -1 * distance
        distance = distance.squeeze(dim=-1)
        return distance