import sys
import os

import torch.nn.functional as F
import torch
import torch.nn as nn
'''
Entity和Relation都可以从头开始学
'''
class KDManager_Reverse_Stastic(nn.Module):
    def __init__(self, args, mode=1):
        super(KDManager_Reverse_Stastic, self).__init__()
        self.args = args
        self.mode = mode
        self.init_size = self.args.init_size
        self.data_type = torch.double if self.args.data_type=='double' else torch.float
        pretrain_model = torch.load(os.path.join(self.args.pretrain_path, 'checkpoint'))
        self.entity_embedding = nn.Parameter(pretrain_model['model_state_dict']['entity_embedding'].cpu(), requires_grad=False)
        self.origin_relation_embedding = nn.Parameter(pretrain_model['model_state_dict']['relation_embedding'].cpu(), requires_grad=False)

        # 欧几里得空间 
        # if mode==1:
        #     self.entity_embedding = nn.Parameter(torch.empty(self.args.nentity, self.args.target_dim*self.args.entity_mul), requires_grad=True)
        # if mode==2:
        #     self.entity_embedding = nn.Parameter(torch.empty(self.args.nentity, self.args.target_dim), requires_grad=True)
            
        # self.relation_embedding = nn.Parameter(torch.empty(self.args.nrelation, self.args.target_dim*self.args.relation_mul), requires_grad=True)
        # nn.init.xavier_uniform_(self.entity_embedding)
        # nn.init.xavier_uniform_(self.relation_embedding)
        
        # AttH & RotH & RefH使用
        if mode == 1:
            self.entity_embedding = nn.Parameter(self.init_size * torch.randn((self.args.nentity, self.args.target_dim * self.args.entity_mul), dtype=self.data_type), requires_grad=True)
        elif mode == 2:
            self.entity_embedding = nn.Parameter(self.init_size * torch.randn((self.args.nentity, self.args.target_dim), dtype=self.data_type), requires_grad=True)

        self.relation_embedding = nn.Parameter(self.init_size * torch.randn((self.args.nrelation, self.args.target_dim * self.args.relation_mul), dtype=self.data_type), requires_grad=True)
        
    
    def forward(self, sample):
        head, tail = self.EntityEmbeddingExtract(self.entity_embedding, sample)
        relation = self.RelationEmbeddingExtract(self.relation_embedding, sample)
        origin_relation = self.RelationEmbeddingExtract(self.origin_relation_embedding, sample)
        return head, relation, tail, origin_relation
    
    def EntityEmbeddingExtract(self, entity_embedding, sample):
        positive, negative = sample
        batch_size, negative_sample_size = negative.size(0), negative.size(1)
        
        neg_tail = torch.index_select(
            entity_embedding, 
            dim=0, 
            index=negative.view(-1)
        ).view(batch_size, negative_sample_size, -1)
        
        pos_tail = torch.index_select(
            entity_embedding, 
            dim=0, 
            index=positive[:, 2]
        ).unsqueeze(1)
        
        tail = torch.cat((pos_tail, neg_tail), dim=1)
        
        head = torch.index_select(
            entity_embedding, 
            dim=0, 
            index=positive[:, 0]
        ).unsqueeze(1)
            
        return head, tail

    def RelationEmbeddingExtract(self, relation_embedding, sample):
        positive, negative = sample
        
        relation = torch.index_select(
                relation_embedding, 
                dim=0, 
                index=positive[:, 1]
            ).unsqueeze(1)
        
        return relation