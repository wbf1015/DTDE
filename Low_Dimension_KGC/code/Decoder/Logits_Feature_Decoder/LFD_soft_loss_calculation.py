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

from LFD_auxiliary_loss import *
from LFD_fusion import *
from LFD_norm import *
from LFD_soft_loss import *
from LFD_similarity import *


"""
========================================软损失计算===================================
"""


def only_learn_right(student_dis, teacher_dis):
    reference_scores = teacher_dis[:, 0].unsqueeze(1)
    
    # Mask for values greater than the reference score
    mask = teacher_dis > reference_scores
    
    # Set values greater than the reference to a large negative value (-1e9)
    modified_teacher_dis = torch.where(mask, torch.tensor(-1e3, device=teacher_dis.device), teacher_dis)
    modified_student_dis = torch.where(mask, torch.tensor(-1e3, device=student_dis.device), student_dis)
    
    return modified_student_dis, modified_teacher_dis

def only_learn_right_mask(student_dis, teacher_dis):
    reference_scores = teacher_dis[:, 0].unsqueeze(1)
    mask = teacher_dis > reference_scores
    
    return mask

class Imitation_SingleTeacher(nn.Module):
    def __init__(self, args, stu_score_preprocess='constant', tea_score_preprocess='constant', distill_func='KL_divergency'):
        super(Imitation_SingleTeacher, self).__init__()
        self.args = args
        self.stu_score_preprocess = globals()[stu_score_preprocess]
        self.tea_score_preprocess = globals()[tea_score_preprocess]
        
        self.distill_func = globals()[distill_func](self.args)
        
    def forward(self, stu_dis, tea_dis, prefix=''):
        stu_dis, _, _ = self.stu_score_preprocess(stu_dis)
        tea_dis, _, _ = self.tea_score_preprocess(tea_dis)

        if type(self.distill_func).__name__ in ['Margin_HuberLoss', 'HuberLoss', 'KL_divergency', 'KL_divergencyv2', 'DistillKL_Logit_Standard']:
            loss = self.distill_func(stu_dis, tea_dis)
        if type(self.distill_func).__name__ in ['']:
            mask = only_learn_right_mask(stu_dis, tea_dis)
            loss = self.distill_func(stu_dis, tea_dis, mask)
        
        loss_record = {
            'soft_loss'+prefix: loss.item(),
        }
        
        return loss, loss_record



class Imitation_DualTeacher(nn.Module):
    def __init__(self, args, stu_score_preprocess='constant', tea_score_preprocess='constant', distill_func='KL_divergency', fusion_function='scores_dmutde_fusion'):
        super(Imitation_DualTeacher, self).__init__()
        self.args = args
        self.stu_score_preprocess = globals()[stu_score_preprocess]
        self.tea_score_preprocess = globals()[tea_score_preprocess]
        
        self.fusion_func = globals()[fusion_function]
        self.distill_func = globals()[distill_func](self.args)
        
            
    def forward(self, stu_dis, tea_dis1, tea_dis2, prefix=''):
        stu_dis, _, _ = self.stu_score_preprocess(stu_dis)
        tea_dis1, _, _ = self.tea_score_preprocess(tea_dis1)
        tea_dis2, _, _ = self.tea_score_preprocess(tea_dis2)

        tea_dis = self.fusion_func(tea_dis1, tea_dis2)
        
        if type(self.distill_func).__name__ in ['Margin_HuberLoss', 'HuberLoss', 'KL_divergency', 'KL_divergencyv2', 'DistillKL_Logit_Standard']:
            loss = self.distill_func(stu_dis, tea_dis)
        if type(self.distill_func).__name__ in ['']:
            mask = only_learn_right_mask(stu_dis, tea_dis)
            loss = self.distill_func(stu_dis, tea_dis, mask)
        
        loss_record = {
            'soft_loss'+prefix: loss.item(),
        }
        
        return loss, loss_record


class Imitation_DualTeacherv2(nn.Module):
    def __init__(self, args, stu_score_preprocess='constant', tea_score_preprocess='constant', distill_func='KL_divergency', fusion_function='scores_dmutde_fusion'):
        super(Imitation_DualTeacherv2, self).__init__()
        self.args = args
        self.stu_score_preprocess = globals()[stu_score_preprocess]
        self.tea_score_preprocess = globals()[tea_score_preprocess]
        
        self.fusion_func = globals()[fusion_function]
        self.distill_func = globals()[distill_func](self.args)
    
    def fusion_score(self, tea_dis1, tea_dis2, weight):
        tea_dis = self.fusion_func(tea_dis1, tea_dis2, weight=weight)
        return tea_dis
            
    def forward(self, stu_dis, tea_dis1, tea_dis2, prefix='', weight=None):
        tea_dis = self.fusion_score(tea_dis1, tea_dis2, weight=weight)
        
        stu_dis, _, _ = self.stu_score_preprocess(stu_dis)
        tea_dis, _, _ = self.tea_score_preprocess(tea_dis)


        if type(self.distill_func).__name__ in ['Margin_HuberLoss', 'HuberLoss', 'KL_divergency', 'KL_divergencyv2','DistillKL_Logit_Standard']:
            loss = self.distill_func(stu_dis, tea_dis)
        if type(self.distill_func).__name__ in ['']:
            mask = only_learn_right_mask(stu_dis, tea_dis)
            loss = self.distill_func(stu_dis, tea_dis, mask)
        
        loss_record = {
            'soft_loss'+prefix: loss.item(),
        }
        
        return loss, loss_record


class Imitation_DualTeacherv3(nn.Module):
    def __init__(self, args, stu_score_preprocess='constant', tea_score_preprocess='constant', distill_func='KL_divergency', fusion_function='scores_dmutde_fusion', fusion_scores_loss='SigmoidLoss'):
        super(Imitation_DualTeacherv3, self).__init__()
        self.args = args
        self.stu_score_preprocess = globals()[stu_score_preprocess]
        self.tea_score_preprocess = globals()[tea_score_preprocess]
        
        self.fusion_func = globals()[fusion_function]
        self.distill_func = globals()[distill_func](self.args)
        self.fusion_scores_loss = globals()[fusion_scores_loss](self.args)
    
    def fusion_score(self, tea_dis1, tea_dis2, weight):
        tea_dis = self.fusion_func(tea_dis1, tea_dis2, weight=weight)
        return tea_dis
            
    def forward(self, stu_dis, tea_dis1, tea_dis2, prefix='', weight=None):
        tea_dis = self.fusion_score(tea_dis1, tea_dis2, weight=weight)
        fusion_score_loss, fusion_score_loss_record = self.fusion_scores_loss(tea_dis, None, big_better=True)
        
        stu_dis, _, _ = self.stu_score_preprocess(stu_dis)
        tea_dis, _, _ = self.tea_score_preprocess(tea_dis)


        if type(self.distill_func).__name__ in ['Margin_HuberLoss', 'HuberLoss', 'KL_divergency', 'KL_divergencyv2','DistillKL_Logit_Standard']:
            loss = self.distill_func(stu_dis, tea_dis)
        if type(self.distill_func).__name__ in ['']:
            mask = only_learn_right_mask(stu_dis, tea_dis)
            loss = self.distill_func(stu_dis, tea_dis, mask)
        
        loss += 1 * fusion_score_loss
        
        loss_record = {
            'soft_loss'+prefix: loss.item()-fusion_score_loss.item(),
            'fusion_score_loss_record'+prefix: fusion_score_loss.item()
        }
        
        return loss, loss_record


