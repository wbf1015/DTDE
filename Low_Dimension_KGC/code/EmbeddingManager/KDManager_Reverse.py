import sys
import os

import torch.nn.functional as F
import torch
import torch.nn as nn
'''
一开始的Manager设置，entity_embedding不需要进行参数更新
'''
class KDManager_Reverse(nn.Module):
    def __init__(self, args):
        super(KDManager_Reverse, self).__init__()
        self.args = args
        pretrain_model = torch.load(os.path.join(self.args.pretrain_path, 'checkpoint'))
        self.entity_embedding = nn.Parameter(pretrain_model['model_state_dict']['entity_embedding'].cpu(), requires_grad=False)
        self.origin_relation_embedding = nn.Parameter(pretrain_model['model_state_dict']['relation_embedding'].cpu(), requires_grad=False)
        # 这里是假定需要从头train关系的embedding
        self.relation_embedding = nn.Parameter(torch.empty(self.args.nrelation, self.args.target_dim*self.args.relation_mul), requires_grad=True)
        nn.init.xavier_uniform_(self.relation_embedding)
    
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