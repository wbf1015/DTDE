import sys
import os

import torch.nn.functional as F
import torch
import torch.nn as nn

class EncoderModel_Reverse(nn.Module):
    def __init__(self, KGE=None, embedding_manager=None, entity_pruner=None, relation_pruner=None, loss=None, kdloss=None, decoder=None, args=None):
        super(EncoderModel_Reverse, self).__init__()
        self.KGE = KGE
        self.EmbeddingManager = embedding_manager
        self.EntityPruner = entity_pruner
        self.RelationPruner = relation_pruner
        self.decoder = decoder
        self.args = args
        self.kdloss = None
        self.ContrastiveLoss = None
            
    def forward(self, data, subsampling_weight):
        head, relation, tail, origin_relation = self.EmbeddingManager(data)
        if self.args.cuda:
            head, relation, tail = head.cuda(), relation.cuda(), tail.cuda()
        head, relation, tail = self.EntityPruner(head), self.RelationPruner(relation), self.EntityPruner(tail)

        loss, loss_record = self.decoder(head, relation, tail, subsampling_weight) # 这个直接换成你想使用的Contastive_LOSS就可以
        return loss, loss_record
    
    def predict(self, data):
        head, relation, tail, _ = self.EmbeddingManager(data)
        head, relation, tail = self.EntityPruner(head), self.RelationPruner(relation), self.EntityPruner(tail)
        score = self.decoder.predict(head, relation, tail)
        return score


''' 配合的是tail一个从头学起 '''
class EncoderModel_Reverse2(nn.Module):
    def __init__(self, KGE=None, embedding_manager=None, entity_pruner=None, relation_pruner=None, loss=None, kdloss=None, decoder=None, args=None):
        super(EncoderModel_Reverse2, self).__init__()
        self.KGE = KGE
        self.EmbeddingManager = embedding_manager
        self.EntityPruner = entity_pruner
        self.RelationPruner = relation_pruner
        self.decoder = decoder
        self.args = args
        self.kdloss = None
        self.ContrastiveLoss = None
        self.mode = 1 # self.mode=1意味着head和relation是分开学习的，反之时一起学习的
        if self.EntityPruner.mode==3 or self.EntityPruner.mode==4:
            self.mode=2
        
            
    def forward(self, data, subsampling_weight):
        head, relation, tail, origin_relation = self.EmbeddingManager(data)
        if self.args.cuda:
            head, relation, tail = head.cuda(), relation.cuda(), tail.cuda()
            
        if self.mode==1:
            head, relation, tail = self.EntityPruner(head), self.RelationPruner(relation), tail
            loss, loss_record = self.decoder(head, relation, tail, subsampling_weight) # 这个直接换成你想使用的Contastive_LOSS就可以
        elif self.mode==2:
            ehr, et = self.EntityPruner(head, origin_relation), tail
            loss, loss_record = self.decoder(ehr, et, subsampling_weight) # 这个直接换成你想使用的Contastive_LOSS就可以
        
        return loss, loss_record
    
    def predict(self, data):
        head, relation, tail, origin_relation = self.EmbeddingManager(data)
        if self.mode==1:
            head, relation, tail = self.EntityPruner(head), self.RelationPruner(relation), tail
            score = self.decoder.predict(head, relation, tail)
        elif self.mode==2:
            ehr, et = self.EntityPruner(head, origin_relation), tail
            
            score = self.decoder.predict(ehr, et)
        return score