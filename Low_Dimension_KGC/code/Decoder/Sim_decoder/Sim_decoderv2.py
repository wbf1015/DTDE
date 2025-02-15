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

class Decoder_2KGEv2(nn.Module):
    def __init__(self, args):
        super(Decoder_2KGEv2, self).__init__()
        self.args = args
        self.target_dim = args.target_dim
        self.entity_dim = args.target_dim * args.entity_mul
        self.relation_dim = args.target_dim * args.relation_mul

        # 硬损失函数
        # self.hard_loss = SigmoidLoss(args=args)
        self.FT_teacher_hardloss = SigmoidLoss2(args=args, pos_margin=2.0, neg_margin=9.0)
        self.hard_loss = SigmoidLoss2(args=args)
        # self.hard_loss = BCELoss(args=args)
        
        self.hard_loss_AT = SigmoidLoss_AT(args=args)
        
        # 软损失函数
        self.FT_teacher_softloss = Imitation_SingleTeacher(args=args, stu_score_preprocess='constant', tea_score_preprocess='constant', distill_func='DistillKL_Logit_Standard')
        self.align_teacher_loss = AlignmentLoss(args=args, tea_score1_preprocess='constant', tea_score2_preprocess='constant')
        self.student_kd_loss = Imitation_DualTeacher(args, stu_score_preprocess='constant', tea_score_preprocess='constant', distill_func='DistillKL_Logit_Standard', fusion_function='scores_dmutde_fusion')
        
        # 辅助模块
        self.cl_module = ContrastiveLoss(args=args)
        self.bias_module = EntityBias(args=args)
        
        # 一些必要的组件
        self.cal_sim = cal_similarity(args=self.args, temperature=self.args.temprature)
        self.Combine_hr = Combine_hr(self.entity_dim, self.relation_dim, self.target_dim, layer_mul=2)
        self.tail_transform = BN(self.target_dim)

    def pre_process_teacherAlign(self, PT1_score, PT2_score, FT1_score, FT2_score):
        # teacher_dis1 = torch.sort(PT1_score + FT1_score, dim=1, descending=True).values
        # teacher_dis2 = torch.sort(PT2_score + FT2_score, dim=1, descending=True).values
        # return teacher_dis1, teacher_dis2
        return PT1_score+FT1_score, PT2_score
    
    def normal_batch(self, eh, er, et, PT1_score, PT2_score, FT1_score, FT2_score, data, subsampling_weight):
        ehr = self.Combine_hr(eh, er)
        et = self.tail_transform(et)
        stu_score = self.cal_sim.SCCF_similarity3(ehr, et)
        
        # Fine-tune teacher model
        FT_teacher1_softloss, FT_teacher1_softloss_record = self.FT_teacher_softloss(PT1_score+FT1_score, PT1_score, prefix='soft_FT_teacher1')
        FT_teacher2_softloss, FT_teacher2_softloss_record = self.FT_teacher_softloss(PT2_score+FT2_score, PT2_score, prefix='soft_FT_teacher2')
        
        FT_teacher1_hardloss, FT_teacher1_hardloss_record = self.FT_teacher_hardloss(PT1_score+FT1_score, subsampling_weight, big_better=True)
        FT_teacher2_hardloss, FT_teacher2_hardloss_record = self.FT_teacher_hardloss(PT2_score+FT2_score, subsampling_weight, big_better=True)
        
        # Align teacher model
        teacher_dis1, teacher_dis2 = self.pre_process_teacherAlign(PT1_score, PT2_score, FT1_score, FT2_score)
        align_teacher_loss, align_teacher_record = self.align_teacher_loss(teacher_dis1, teacher_dis2, prefix='Align Teacher')
        
        # train_student_model
        hard_loss, loss_record = self.hard_loss(stu_score, subsampling_weight, big_better=True)
        soft_loss, soft_loss_record = self.student_kd_loss(stu_score, PT1_score+FT1_score, PT2_score, prefix='')
    
        # 对比学习损失函数
        # contrastive_loss, contrastive_loss_record = self.cl_module(stu_score2, None)
        # contrastive_loss, contrastive_loss_record = self.cl_module.forward2(ehr, et)
        
        loss = hard_loss
        loss += self.args.kdloss_weight * soft_loss
        loss += 1 * FT_teacher1_softloss
        loss += 0 * FT_teacher2_softloss
        loss += 1 * FT_teacher1_hardloss
        loss += 0 * FT_teacher2_hardloss
        loss += 0.1 * align_teacher_loss
        # loss += self.args.contrastive_weight * contrastive_loss
        
        loss_record.update(soft_loss_record)
        loss_record.update(FT_teacher1_softloss_record)
        loss_record.update(FT_teacher2_softloss_record)
        loss_record.update(FT_teacher1_hardloss_record)
        loss_record.update(FT_teacher2_hardloss_record)
        loss_record.update(align_teacher_record)
        # loss_record.update(contrastive_loss_record)
        loss_record.update({'LOSS':loss.item()})
        
        return loss, loss_record
    
    def all_true_batch(self, eh, er, et, PT1_score, PT2_score, FT1_score, FT2_score, data, subsampling_weight):
        ehr = self.Combine_hr(eh, er)
        et = self.tail_transform(et)

        stu_score = self.cal_sim.SCCF_similarity3(ehr, et)
        hard_loss, loss_record = self.hard_loss_AT((ehr, et), stu_score, data=data, subsampling_weight=subsampling_weight, big_better=True)
        
        loss = hard_loss
        
        return loss, loss_record
    
    def forward(self, eh, er, et, PT1_score, PT2_score, FT1_score, FT2_score, data, subsampling_weight, mode):
        if mode=='1posNneg':
            loss, loss_record = self.normal_batch(eh, er, et, PT1_score, PT2_score, FT1_score, FT2_score, data, subsampling_weight)
        
        if mode=='alltrue':
            loss, loss_record =self.all_true_batch(eh, er, et, PT1_score, PT2_score, FT1_score, FT2_score, data, subsampling_weight)
        
        return loss, loss_record
    
    def predict(self, eh, er, et, PT1_score, PT2_score, FT1_score, FT2_score, data):
        ehr = self.Combine_hr(eh, er)
        et = self.tail_transform(et)

        stu_score = self.cal_sim.SCCF_similarity3(ehr, et)
        return stu_score
    
    def predict2(self, eh, er, et, PT1_score, PT2_score, FT1_score, FT2_score, data):
        score = PT1_score + FT1_score
        
        return score

    def predict3(self, eh, er, et, PT1_score, PT2_score, FT1_score, FT2_score, data):
        score = PT2_score + FT2_score
        
        return score


