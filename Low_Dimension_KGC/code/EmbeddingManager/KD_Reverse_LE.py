import sys
import os

import torch.nn.functional as F
import torch
import torch.nn as nn

'''
提供了更多的学习知识的方案：每一个entity还拥有一个独立的，从头学习的嵌入（LE：Learning&Extraction）
也就是说，从高维模型中的学习和从头学习中并举
'''
class KD_Reverse_LE(nn.Module):
    def __init__(self, args, mode=1):
        super(KD_Reverse_LE, self).__init__()
        self.args = args
        pretrain_model = torch.load(os.path.join(self.args.pretrain_path, 'checkpoint'))
        self.entity_embedding = nn.Parameter(pretrain_model['model_state_dict']['entity_embedding'].cpu(), requires_grad=False)
        self.origin_relation_embedding = nn.Parameter(pretrain_model['model_state_dict']['relation_embedding'].cpu(), requires_grad=False)
        
        self.relation_embedding = nn.Parameter(torch.empty(self.args.nrelation, self.args.target_dim*self.args.relation_mul), requires_grad=True) # 这里是假定需要从头train关系的嵌入
        if mode==1:
            self.entity_embedding_FS = nn.Parameter(torch.empty(self.args.nentity, self.args.target_dim*self.args.entity_mul), requires_grad=True)         # 然后这里也有一个需要从头开始train的实体的嵌入
        elif mode==2:
            self.entity_embedding_FS = nn.Parameter(torch.empty(self.args.nentity, self.args.target_dim), requires_grad=True)         # 然后这里也有一个需要从头开始train的实体的嵌入
            
        nn.init.xavier_uniform_(self.relation_embedding)
        nn.init.xavier_uniform_(self.entity_embedding_FS)
    
    def forward(self, sample):
        head, tail = self.EntityEmbeddingExtract(self.entity_embedding, self.entity_embedding_FS, sample)
        relation = self.RelationEmbeddingExtract(self.relation_embedding, sample)
        origin_relation = self.RelationEmbeddingExtract(self.origin_relation_embedding, sample)
        return head, relation, tail, origin_relation
    
    def EntityEmbeddingExtract(self, entity_embedding, entity_embedding_FS, sample):
        positive, negative = sample
        batch_size, negative_sample_size = negative.size(0), negative.size(1)
        
        # FS：From_scratch
        
        neg_tail = torch.index_select(
            entity_embedding_FS, 
            dim=0, 
            index=negative.view(-1)
        ).view(batch_size, negative_sample_size, -1)
        
        pos_tail = torch.index_select(
            entity_embedding_FS, 
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