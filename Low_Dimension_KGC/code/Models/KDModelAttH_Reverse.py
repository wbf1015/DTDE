import sys
import os

import torch.nn.functional as F
import torch
import torch.nn as nn

class KDModelAttH_Reverse(nn.Module):
    def __init__(self, KGE=None, embedding_manager=None, entity_pruner=None, relation_pruner=None, loss=None, kdloss=None, decoder=None, args=None):
        super(KDModelAttH_Reverse, self).__init__()
        self.KGE = KGE
        self.EmbeddingManager = embedding_manager
        self.EntityPruner = entity_pruner
        self.RelationPruner = relation_pruner
        self.loss = loss
        self.KDLoss = kdloss
        self.KDLossWeight = args.kdloss_weight
        self.args = args
        self.data_type = torch.double if self.args.data_type=='double' else torch.float
    
    def get_postive_score(self, score):
        return score[:, 0]
    
    def get_negative_score(self, score):
        return score[:, 1:]
    
    def forward(self, data, subsampling_weight):
        head, relation, tail, origin_relation = self.EmbeddingManager(data)
        if self.args.cuda:
            head, relation, tail, origin_relation = head.cuda(), relation.cuda(), tail.cuda(), origin_relation.cuda()
        score = self.KGE(head, relation, tail, data)
        p_score, n_score = self.get_postive_score(score), self.get_negative_score(score)
        loss, loss_record = self.loss(p_score, n_score, subsampling_weight)
        
        return loss, loss_record
    
    def predict(self, data):
        head, relation, tail, origin_relation = self.EmbeddingManager(data)
        score = self.KGE(head, relation, tail, data)
        return score

    def set_kdloss(self,kdloss):
        self.KDLoss = kdloss