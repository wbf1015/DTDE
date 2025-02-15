import sys
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

original_directory = os.getcwd()
new_directory = original_directory + '/code/Decoder/Sim_decoder/'
if new_directory not in sys.path:
    sys.path.append(new_directory)

from SD_auxiliary_loss import *
from SD_contrastive_learning import *
from SD_fusion import *
from SD_norm import *
from SD_soft_loss import *
from SD_hard_loss import *
from SD_embedding_transform import *
from SD_soft_loss_calculation import *
from SD_similarity import *

sys.path.remove(new_directory)

"""

=======================================================对外接口类============================================

"""

class Decoder_2KGE(nn.Module):
    def __init__(self, args):
        super(Decoder_2KGE, self).__init__()
        self.args = args
        self.target_dim = args.target_dim
        self.entity_dim = args.target_dim * args.entity_mul
        self.relation_dim = args.target_dim * args.relation_mul

        # 硬损失函数
        self.hard_loss1 = SigmoidLoss(args=args)
        self.hard_loss2 = SigmoidLoss2(args=args)
        # self.hard_loss = BCELoss(args=args)
        self.hard_loss_AT = SigmoidLoss_AT(args=args)
        
        # 软损失函数
        # self.soft_loss = DistanceImitation_SingleTeacher(args=args, stu_score_preprocess='local_standardize', tea_score_preprocess='constant', distill_func='KL_divergency')
        self.soft_loss = Imitation_SingleTeacher(args=args, stu_score_preprocess='local_standardize', tea_score_preprocess='constant', distill_func='KL_divergency')
        # self.soft_loss2 = Imitation_SingleTeacher(args=args, stu_score_preprocess='constant', tea_score_preprocess='constant', distill_func='DistillKL_Logit_Standard')
        # self.soft_loss2 = Imitation_DualTeacherv2(args=args, stu_score_preprocess='local_standardize', tea_score_preprocess='constant', distill_func='KL_divergency', fusion_function='scores_fusionv1')
        self.soft_loss2 = Imitation_DualTeacherv3(args=args, stu_score_preprocess='local_standardize', tea_score_preprocess='constant', distill_func='KL_divergency', fusion_function='scores_fusionv1', fusion_scores_loss='SigmoidLoss2')
        
        # 辅助模块
        self.cl_module = ContrastiveLossv2(args=args)
        # self.bias_module = EntityBias(args=args)
        self.weight_learner = weight_learnerv2(args)
        self.Knowledge_Injection = Knowledge_Injection(args)
        
        # 一些必要的组件
        self.cal_sim = cal_similarity(args=self.args, temperature=self.args.temprature)
        self.Combine_hr = Combine_hr(self.entity_dim, self.relation_dim, self.target_dim, layer_mul=2)
        self.tail_transform = BN(self.target_dim)


    def get_student_score(self, ehr, et):        
        stu_score = self.cal_sim.SCCF_similarity3(ehr, et)
        stu_score[:, 0] -= self.args.pos_gamma  # 第一列减去pos_gamma
        stu_score[:, 1:] -= self.args.neg_gamma  # 其余列减去neg_gamma
        
        return stu_score
        
    def random_sample_batch(self, eh, er, et, PT1_score, PT2_score, data=None, Teacher_embeddings=None, subsampling_weight=None):
        ehr = self.Combine_hr(eh, er)
        et = self.tail_transform(et)
        
        stu_score= self.get_student_score(ehr, et)
        
        weight = self.weight_learner(ehr, et, data)
        
        hard_loss, loss_record = self.hard_loss1(stu_score, subsampling_weight, big_better=True)
        hard_loss2, loss_record2 = self.hard_loss2(stu_score, subsampling_weight, big_better=True)
        
        soft_loss1, soft_loss_record1 = self.soft_loss(stu_score, PT1_score, prefix='RotatE')
        soft_loss2, soft_loss_record2 = self.soft_loss(stu_score, PT2_score, prefix='LorentzKG')
        soft_lossfusion, soft_loss_recordfusion = self.soft_loss2(stu_score, PT1_score, PT2_score, prefix='Fusion', weight=weight)
        
        # kiloss, kiloss_record = self.Knowledge_Injection(ehr, et, PT1_score, PT2_score)
        # contrastive_loss, contrastive_loss_record = self.cl_module(stu_score2, None)
        # contrastive_loss, contrastive_loss_record = self.cl_module.forward2(ehr, et)
        
        loss = 0 * hard_loss
        loss += 1 * hard_loss2
        loss += 0.0 * self.args.kdloss_weight * soft_loss1 
        loss += 1 * self.args.kdloss_weight * soft_loss2
        loss += 1 * self.args.kdloss_weight * soft_lossfusion
        # loss += self.args.contrastive_weight * contrastive_loss
        # loss += 0.1 * kiloss
        
        loss_record.update(loss_record2)
        loss_record.update(soft_loss_record1)
        loss_record.update(soft_loss_record2)
        loss_record.update(soft_loss_recordfusion)
        # loss_record.update(contrastive_loss_record)
        # loss_record.update(kiloss_record)
        loss_record.update({'LOSS':loss.item()})
        
        return loss, loss_record
    
    def query_sample_batch(self, eh, er, et, PT1_score, PT2_score, data=None, Teacher_embeddings=None, subsampling_weight=None):
        ehr = self.Combine_hr(eh, er)
        et = self.tail_transform(et)
        
        stu_score= self.get_student_score(ehr, et)
        
        weight = self.weight_learner(ehr, et, data)
        
        hard_loss, loss_record = self.hard_loss1(stu_score, subsampling_weight)
        hard_loss2, loss_record2 = self.hard_loss2(stu_score, subsampling_weight)
        
        soft_loss1, soft_loss_record1 = self.soft_loss(stu_score, PT1_score, prefix='RotatE')
        soft_loss2, soft_loss_record2 = self.soft_loss(stu_score, PT2_score, prefix='LorentzKG')
        soft_lossfusion, soft_loss_recordfusion = self.soft_loss2(stu_score, PT1_score, PT2_score, prefix='Fusion', weight=weight)
        
        # kiloss, kiloss_record = self.Knowledge_Injection(ehr, et, PT1_score, PT2_score)
        # contrastive_loss, contrastive_loss_record = self.cl_module(eh, er, et, Teacher_embeddings)
        
        
        loss = 0 * hard_loss
        loss += 1 * hard_loss2
        loss += 0.0 * self.args.kdloss_weight * soft_loss1 
        loss += 1 * self.args.kdloss_weight * soft_loss2
        loss += 0.5 * self.args.kdloss_weight * soft_lossfusion
        # loss += 0.1 * kiloss
        # loss += self.args.contrastive_weight * contrastive_loss

        
        loss_record.update(loss_record2)
        loss_record.update(soft_loss_record1)
        loss_record.update(soft_loss_record2)
        loss_record.update(soft_loss_recordfusion)
        # loss_record.update(kiloss_record)
        # loss_record.update(contrastive_loss_record)
        loss_record.update({'LOSS':loss.item()})
        
        return loss, loss_record
    
    def all_true_batch(self, eh, er, et, PT1_score, PT2_score, data, subsampling_weight=None):
        ehr = self.Combine_hr(eh, er)
        et = self.tail_transform(et)

        stu_score = self.cal_sim.SCCF_similarity3(ehr, et)
        hard_loss, loss_record = self.hard_loss_AT((ehr, et), stu_score, data=data, subsampling_weight=subsampling_weight, big_better=True)
        
        loss = hard_loss
        
        return loss, loss_record
    
    def forward(self, eh, er, et, PT1_score, PT2_score, data=None, Teacher_embeddings=None, subsampling_weight=None, mode='1posNneg'):
        if mode=='RandomSample':
            loss, loss_record = self.random_sample_batch(eh, er, et, PT1_score, PT2_score, data=data, Teacher_embeddings=Teacher_embeddings, subsampling_weight=subsampling_weight)
        
        if mode=='QueryAwareSample':
            loss, loss_record = self.query_sample_batch(eh, er, et, PT1_score, PT2_score, data=data, Teacher_embeddings=Teacher_embeddings, subsampling_weight=subsampling_weight)
        
        if mode=='alltrue':
            loss, loss_record =self.all_true_batch(eh, er, et, PT1_score, PT2_score, data=data, subsampling_weight=subsampling_weight)
        
        return loss, loss_record
    
    def predict(self, eh, er, et, PT1_score, PT2_score):
        score = self.soft_loss2.fusion_score(PT1_score, PT2_score)
        
        return score
    
    def predict2(self, eh, er, et):
        ehr = self.Combine_hr(eh, er)
        et = self.tail_transform(et)

        stu_score = self.cal_sim.SCCF_similarity3(ehr, et)
        return stu_score

    def predict3(self, eh, er, et):
        ehr = self.Combine_hr(eh, er)
        et = self.tail_transform(et)
        stu_score = self.cal_sim.SCCF_similarity3(ehr, et)
        stu_score2 = self.soft_loss.get_stu_dis(eh, er, et, ehr, et, None)
        stu_score += stu_score2
        return stu_score


