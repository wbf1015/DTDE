import sys
import os

import torch.nn.functional as F
import torch
import torch.nn as nn
'''
允许读入两个预训练的模型，因为学习到的embedding是在另一个空间的，所以没关系，entity和relation都是从头学习的
'''
class KDManager_Reverse_2KGEv2(nn.Module):
    def __init__(self, args):
        super(KDManager_Reverse_2KGEv2, self).__init__()
        self.args = args
        self.data_type = torch.double if self.args.data_type == 'double' else torch.float
        pretrain_model = torch.load(os.path.join(self.args.pretrain_path, 'checkpoint'))

        # 一般我们默认第一个pretrain_path内容的是欧式空间的embedding
        if 'RotatE' in self.args.pretrain_path:
            if 'entity_embedding' in pretrain_model['model_state_dict']:
                self.PT_entity_embedding = nn.Parameter(pretrain_model['model_state_dict']['entity_embedding'].cpu().to(self.data_type), requires_grad=False)
                self.PT_relation_embedding = nn.Parameter(pretrain_model['model_state_dict']['relation_embedding'].cpu().to(self.data_type), requires_grad=False)
            else:
                self.PT_entity_embedding = nn.Parameter(pretrain_model['model_state_dict']['EmbeddingManager.entity_embedding'].cpu().to(self.data_type), requires_grad=False)
                self.PT_relation_embedding = nn.Parameter(pretrain_model['model_state_dict']['EmbeddingManager.relation_embedding'].cpu().to(self.data_type), requires_grad=False)
        if 'AttH' in self.args.pretrain_path:
            self.PT_entity_embedding = nn.Parameter(pretrain_model['model_state_dict']['EmbeddingManager.entity_embedding'].cpu().to(self.data_type), requires_grad=False)
            self.PT_relation_embedding = nn.Parameter(pretrain_model['model_state_dict']['EmbeddingManager.relation_embedding'].cpu().to(self.data_type), requires_grad=False)
        
        
        if 'RotatE' in self.args.pretrain_path2:
            pretrain_model = torch.load(os.path.join(self.args.pretrain_path2, 'checkpoint'))
            if 'entity_embedding' in pretrain_model['model_state_dict']:
                self.PT_entity_embedding2 = nn.Parameter(pretrain_model['model_state_dict']['entity_embedding'].cpu().to(self.data_type), requires_grad=False)
                self.PT_relation_embedding2 = nn.Parameter(pretrain_model['model_state_dict']['relation_embedding'].cpu().to(self.data_type), requires_grad=False)
            else:
                self.PT_entity_embedding2 = nn.Parameter(pretrain_model['model_state_dict']['EmbeddingManager.entity_embedding'].cpu().to(self.data_type), requires_grad=False)
                self.PT_relation_embedding2 = nn.Parameter(pretrain_model['model_state_dict']['EmbeddingManager.relation_embedding'].cpu().to(self.data_type), requires_grad=False)
        elif 'LorentzKG' in self.args.pretrain_path2:
            self.PT_entity_embedding2 = None
            self.PT_relation_embedding2 = None
        elif 'AttH' in self.args.pretrain_path2: 
            pretrain_model = torch.load(os.path.join(self.args.pretrain_path2, 'checkpoint'))
            self.PT_entity_embedding2 = nn.Parameter(pretrain_model['model_state_dict']['EmbeddingManager.entity_embedding'].cpu().to(self.data_type), requires_grad=False)
            self.PT_relation_embedding2 = nn.Parameter(pretrain_model['model_state_dict']['EmbeddingManager.relation_embedding'].cpu().to(self.data_type), requires_grad=False)

        self.FT_embedding_dim = 32
        self.FT_entity_embedding1 = nn.Parameter(torch.empty(self.args.nentity, self.FT_embedding_dim * 2, dtype=self.data_type), requires_grad=True)
        self.FT_relation_embedding1 = nn.Parameter(torch.empty(self.args.nrelation, self.FT_embedding_dim * 1, dtype=self.data_type), requires_grad=True)
        self.FT_entity_embedding2 = nn.Parameter(torch.empty(self.args.nentity, self.FT_embedding_dim * 2, dtype=self.data_type), requires_grad=True)
        self.FT_relation_embedding2 = nn.Parameter(torch.empty(self.args.nrelation, self.FT_embedding_dim * 1, dtype=self.data_type), requires_grad=True)
        nn.init.xavier_uniform_(self.FT_entity_embedding1)
        nn.init.xavier_uniform_(self.FT_relation_embedding1)
        nn.init.xavier_uniform_(self.FT_entity_embedding2)
        nn.init.xavier_uniform_(self.FT_relation_embedding2)
        
        # self.FT_entity_embedding1 = nn.Parameter(self.PT_entity_embedding[:,:self.FT_embedding_dim * 2].clone(), requires_grad=True)
        # self.FT_relation_embedding1 = nn.Parameter(self.PT_relation_embedding[:,:self.FT_embedding_dim * 1].clone(), requires_grad=True)
        # if self.PT_entity_embedding2 is not None:
        #     self.FT_entity_embedding2 = nn.Parameter(self.PT_entity_embedding2[:,:self.FT_embedding_dim * 2].clone(), requires_grad=True)
        # if self.PT_relation_embedding2 is not None:
        #     self.FT_relation_embedding2 = nn.Parameter(self.PT_relation_embedding2[:,:self.FT_embedding_dim * 1].clone(), requires_grad=True)


        # 这里是假定需要从头train关系的embedding
        self.entity_embedding = nn.Parameter(torch.empty(self.args.nentity, self.args.target_dim * self.args.entity_mul, dtype=self.data_type), requires_grad=True)
        self.relation_embedding = nn.Parameter(torch.empty(self.args.nrelation, self.args.target_dim * self.args.relation_mul, dtype=self.data_type), requires_grad=True)
        nn.init.xavier_uniform_(self.relation_embedding)
        nn.init.xavier_uniform_(self.entity_embedding)

    
    def forward(self, sample):
        PT_head1, PT_tail1 = self.EntityEmbeddingExtract(self.PT_entity_embedding, sample)
        PT_head2, PT_tail2 = self.EntityEmbeddingExtract(self.PT_entity_embedding2, sample)
        head, tail = self.EntityEmbeddingExtract(self.entity_embedding, sample)
        
        PT_relation1 = self.RelationEmbeddingExtract(self.PT_relation_embedding, sample)
        PT_relation2 = self.RelationEmbeddingExtract(self.PT_relation_embedding2, sample)
        relation = self.RelationEmbeddingExtract(self.relation_embedding, sample)
        
        return head, relation, tail, PT_head1, PT_relation1, PT_tail1, PT_head2, PT_relation2, PT_tail2
    
    
    def forward_(self, sample):
        FT_head1, FT_tail1 = self.EntityEmbeddingExtract(self.FT_entity_embedding1, sample)
        FT_head2, FT_tail2 = self.EntityEmbeddingExtract(self.FT_entity_embedding2, sample)
        
        FT_relation1 = self.RelationEmbeddingExtract(self.FT_relation_embedding1, sample)
        FT_relation2 = self.RelationEmbeddingExtract(self.FT_relation_embedding2, sample)
        
        return FT_head1, FT_relation1, FT_tail1, FT_head2, FT_relation2, FT_tail2
        
    
    def EntityEmbeddingExtract(self, entity_embedding, sample):
        
        if entity_embedding is None:
            return None, None
        
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
        
        if relation_embedding is None:
            return None
        
        positive, negative = sample
        
        relation = torch.index_select(
                relation_embedding, 
                dim=0, 
                index=positive[:, 1]
            ).unsqueeze(1)
        
        return relation