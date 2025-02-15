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
from SD_fusion import *
from SD_norm import *
from SD_soft_loss import *
from SD_similarity import *


"""
========================================软损失计算===================================
"""

class cal_softloss_FusionLoss(nn.Module):
    def __init__(self, args, stu_score_preprocess='constant', PT1_score_preprocess='constant', PT2_score_preprocess='constant', distill_func='KL_divergency', fusion_func='loss_half_fusion'):
        super(cal_softloss_FusionLoss, self).__init__()
        self.args = args
        self.stu_score_preprocess = globals()[stu_score_preprocess]
        self.PT1_score_preprocess = globals()[PT1_score_preprocess]
        self.PT2_score_preprocess = globals()[PT2_score_preprocess]
        
        if distill_func == 'HuberLoss':
            self.distill_func = globals()[distill_func](self.args)
        if distill_func == 'KL_divergency':
            self.distill_func = globals()[distill_func](self.args)
        if distill_func == 'KL_divergency2':
            self.distill_func = globals()[distill_func](self.args)
        
        self.fusion_func = globals()[fusion_func]
        
    def forward(self, stu_dis, PT1_dis, PT2_dis):
        stu_dis, _, _ = self.stu_score_preprocess(stu_dis)
        PT1_dis, _, _ = self.PT1_score_preprocess(PT1_dis)
        PT2_dis, _, _ = self.PT2_score_preprocess(PT2_dis)

        if type(self.distill_func).__name__ in ['HuberLoss', 'KL_divergency']:
            loss1 = self.distill_func(stu_dis, PT1_dis)
            loss2 = self.distill_func(stu_dis, PT2_dis)
        if type(self.distill_func).__name__ in ['KL_divergency2']:
            mask = only_learn_right_mask(stu_dis, PT1_dis)
            loss1 = self.distill_func(stu_dis, PT1_dis, mask)
            mask = only_learn_right_mask(stu_dis, PT2_dis)
            loss2 = self.distill_func(stu_dis, PT2_dis, mask)
        
        soft_loss = self.fusion_func(loss1, loss2)
        
        loss_record = {
            'soft_loss_PT1': loss1.item(),
            'soft_loss_PT2': loss2.item(),
            'soft_loss': soft_loss.item(),
        }
        
        return soft_loss, loss_record

class cal_softloss_FusionScores(nn.Module):
    def __init__(self, args, stu_score_preprocess='constant', PT1_score_preprocess='constant', PT2_score_preprocess='constant', distill_func='KL_divergency', fusion_func='scores_add_fusion'):
        super(cal_softloss_FusionScores, self).__init__()
        self.args = args
        self.stu_score_preprocess = globals()[stu_score_preprocess]
        self.PT1_score_preprocess = globals()[PT1_score_preprocess]
        self.PT2_score_preprocess = globals()[PT2_score_preprocess]
        
        if distill_func == 'HuberLoss':
            self.distill_func = globals()[distill_func](self.args)
        if distill_func == 'KL_divergency':
            self.distill_func = globals()[distill_func](self.args)
        
        self.fusion_func = globals()[fusion_func]
    
    def get_betterPT_score(self, stu_dis, PT1_dis, PT2_dis):
        stu_dis_norm, stu_mean, stu_std = local_standardize(stu_dis)
        PT1_dis_norm, PT1_mean, PT1_std = local_standardize(PT1_dis)
        PT2_dis_norm, PT2_mean, PT2_std = local_standardize(PT2_dis)
        
        PT_score = self.fusion_func(PT1_dis_norm, PT2_dis_norm)
        # PT_score = inverse_local_standardize(PT_score, PT2_mean, PT2_std)
        
        # PT_score = self.PT_scale * PT_score
        
        return PT_score
        
    
    def forward(self, stu_dis, PT1_dis, PT2_dis):
        stu_dis_norm, _, _ = self.stu_score_preprocess(stu_dis)
        PT1_dis_norm, _, _ = self.PT1_score_preprocess(PT1_dis)
        PT2_dis_norm, _, _ = self.PT2_score_preprocess(PT2_dis)
        
        PT_score = self.get_betterPT_score(stu_dis, PT1_dis, PT2_dis)
        
        # record_rank_4_case_study2(PT1_dis_norm, PT2_dis_norm, PT_score, 'rank2.txt')
        # sys.exit(0)
        soft_loss = self.distill_func(stu_dis_norm, PT_score)
        
        loss_record = {
            'soft_loss': soft_loss.item(),
        }
        
        return soft_loss, loss_record


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


def record_rank_4_case_study(PT1_dis, PT2_dis, PT_dis, record_path):
    with open(record_path, "w") as f:
        # Iterate through each batch
        for batch_idx in range(PT1_dis.size(0)):
            # Get the current batch scores
            PT1_scores = PT1_dis[batch_idx]
            PT2_scores = PT2_dis[batch_idx]
            PT_scores = PT_dis[batch_idx]

            # Get the indices of the sorted scores (from high to low)
            PT1_sorted_indices = torch.argsort(PT1_scores, descending=True)
            PT2_sorted_indices = torch.argsort(PT2_scores, descending=True)
            PT_sorted_indices = torch.argsort(PT_scores, descending=True)

            # Sort the scores based on the indices
            PT1_sorted_scores = PT1_scores[PT1_sorted_indices]
            PT2_sorted_scores = PT2_scores[PT2_sorted_indices]
            PT_sorted_scores = PT_scores[PT_sorted_indices]

            # Apply softmax to the sorted scores
            PT1_softmax = F.softmax(PT1_sorted_scores, dim=0)
            PT2_softmax = F.softmax(PT2_sorted_scores, dim=0)
            PT_softmax = F.softmax(PT_sorted_scores, dim=0)

            # Write the sorted indices, original scores, and softmax values to the file
            f.write("PT1 sorted indices: " + " ".join(map(str, PT1_sorted_indices.tolist())) + "\n")
            f.write("PT1 original scores: " + " ".join(map(lambda x: f"{x:.4f}", PT1_sorted_scores.tolist())) + "\n")
            f.write("PT1 softmax values: " + " ".join(map(lambda x: f"{x:.4f}", PT1_softmax.tolist())) + "\n")
            
            f.write("PT2 sorted indices: " + " ".join(map(str, PT2_sorted_indices.tolist())) + "\n")
            f.write("PT2 original scores: " + " ".join(map(lambda x: f"{x:.4f}", PT2_sorted_scores.tolist())) + "\n")
            f.write("PT2 softmax values: " + " ".join(map(lambda x: f"{x:.4f}", PT2_softmax.tolist())) + "\n")
            
            f.write("PT sorted indices: " + " ".join(map(str, PT_sorted_indices.tolist())) + "\n")
            f.write("PT original scores: " + " ".join(map(lambda x: f"{x:.4f}", PT_sorted_scores.tolist())) + "\n")
            f.write("PT softmax values: " + " ".join(map(lambda x: f"{x:.4f}", PT_softmax.tolist())) + "\n")

            # Add two blank lines after each batch
            f.write("\n\n")

def record_rank_4_case_study2(PT1_dis, PT2_dis, PT_dis, record_path):
    with open(record_path, "w") as f:
        # Iterate through each batch
        for batch_idx in range(PT1_dis.size(0)):
            # Get the current batch scores
            PT1_scores = PT1_dis[batch_idx]
            PT2_scores = PT2_dis[batch_idx]
            PT_scores = PT_dis[batch_idx]

            # Get the indices of the sorted scores (from high to low)
            PT1_sorted_indices = torch.argsort(PT1_scores, descending=True)
            PT2_sorted_indices = torch.argsort(PT2_scores, descending=True)
            PT_sorted_indices = torch.argsort(PT_scores, descending=True)

            # Sort the scores based on the indices
            PT1_sorted_scores = PT1_scores[PT1_sorted_indices]
            PT2_sorted_scores = PT2_scores[PT2_sorted_indices]
            PT_sorted_scores = PT_scores[PT_sorted_indices]

            PT1_sorted_scores_relative = -1 * ((PT1_sorted_scores[0] - PT1_sorted_scores) / torch.abs(PT1_sorted_scores[0] + 1e-8))
            PT2_sorted_scores_relative = -1 * ((PT2_sorted_scores[0] - PT2_sorted_scores) / torch.abs(PT2_sorted_scores[0] + 1e-8))
            PT_sorted_scores_relative = -1 * ((PT_sorted_scores[0] - PT_sorted_scores) / torch.abs(PT_sorted_scores[0] + 1e-8))

            PT1_softmax = F.softmax(PT1_sorted_scores, dim=0)
            PT2_softmax = F.softmax(PT2_sorted_scores, dim=0)
            PT_softmax = F.softmax(PT_sorted_scores, dim=0)

            PT1_softmax_relative = F.softmax(PT1_sorted_scores_relative, dim=0)
            PT2_softmax_relative = F.softmax(PT2_sorted_scores_relative, dim=0)
            PT_softmax_relative = F.softmax(PT_sorted_scores_relative, dim=0)

            # Write the sorted indices, original scores, relative scores, and softmax values to the file
            f.write("PT1 sorted indices: " + " ".join(map(str, PT1_sorted_indices.tolist())) + "\n")
            f.write("PT1 original scores: " + " ".join(map(lambda x: f"{x:.4f}", PT1_sorted_scores.tolist())) + "\n")
            f.write("PT1 relative scores: " + " ".join(map(lambda x: f"{x:.4f}", PT1_sorted_scores_relative.squeeze().tolist())) + "\n")
            f.write("PT1 softmax values: " + " ".join(map(lambda x: f"{x:.4f}", PT1_softmax.tolist())) + "\n")
            f.write("PT1 softmax relative values: " + " ".join(map(lambda x: f"{x:.4f}", PT1_softmax_relative.tolist())) + "\n")
            f.write("\n")
            
            f.write("PT2 sorted indices: " + " ".join(map(str, PT2_sorted_indices.tolist())) + "\n")
            f.write("PT2 original scores: " + " ".join(map(lambda x: f"{x:.4f}", PT2_sorted_scores.tolist())) + "\n")
            f.write("PT2 relative scores: " + " ".join(map(lambda x: f"{x:.4f}", PT2_sorted_scores_relative.squeeze().tolist())) + "\n")
            f.write("PT2 softmax values: " + " ".join(map(lambda x: f"{x:.4f}", PT2_softmax.tolist())) + "\n")
            f.write("PT2 softmax relative values: " + " ".join(map(lambda x: f"{x:.4f}", PT2_softmax_relative.tolist())) + "\n")
            f.write("\n")
            
            f.write("PT sorted indices: " + " ".join(map(str, PT_sorted_indices.tolist())) + "\n")
            f.write("PT original scores: " + " ".join(map(lambda x: f"{x:.4f}", PT_sorted_scores.tolist())) + "\n")
            f.write("PT relative scores: " + " ".join(map(lambda x: f"{x:.4f}", PT_sorted_scores_relative.squeeze().tolist())) + "\n")
            f.write("PT softmax values: " + " ".join(map(lambda x: f"{x:.4f}", PT_softmax.tolist())) + "\n")
            f.write("PT softmax relative values: " + " ".join(map(lambda x: f"{x:.4f}", PT_softmax_relative.tolist())) + "\n")

            # Add two blank lines after each batch
            f.write("\n\n\n")
            

class DistanceImitation_SingleTeacher(nn.Module):
    def __init__(self, args, stu_score_preprocess='constant', tea_score_preprocess='constant', distill_func='KL_divergency'):
        super(DistanceImitation_SingleTeacher, self).__init__()
        self.args = args
        self.target_dim = args.target_dim
        self.entity_dim = args.target_dim * args.entity_mul
        self.relation_dim = args.target_dim * args.relation_mul
        self.layer_mul = 2
        
        self.cal_sim = cal_similarity(args=self.args, temperature=self.args.temprature)
        
        self.stu_score_preprocess = globals()[stu_score_preprocess]
        self.tea_score_preprocess = globals()[tea_score_preprocess]
        
        if distill_func == 'HuberLoss':
            self.distill_func = globals()[distill_func](self.args)
        if distill_func == 'KL_divergency':
            self.distill_func = globals()[distill_func](self.args)
        if distill_func == 'KL_divergency2':
            self.distill_func = globals()[distill_func](self.args)
        
        self.query_scale = nn.Sequential(
            nn.Linear((2*self.entity_dim + self.relation_dim), (2*self.entity_dim + self.relation_dim) * self.layer_mul),
            nn.LeakyReLU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear((2*self.entity_dim + self.relation_dim) * self.layer_mul, 1)
        )
        
        self.tail_scale = nn.Sequential(
            nn.Linear((self.entity_dim), (self.entity_dim) * self.layer_mul),
            nn.LeakyReLU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear((self.entity_dim) * self.layer_mul, 1)
        )
        
    
    def get_stu_dis(self, eh, er, et, ehr_, et_, data):
        # eh, er, et, ehr_, et_ = eh.detach(), er.detach(), et.detach(), ehr_.detach(), et_.detach()
        ehr_scale = self.query_scale(torch.cat((eh,er,ehr_),dim=-1))
        et_scale = self.tail_scale(et_)
        ehr_ = ehr_ * ehr_scale
        et_ = et_ * et_scale
        transe = et_-ehr_
        stu_dis = torch.norm(transe, p=2, dim=-1)
        stu_dis = -1*stu_dis
        return stu_dis
        
    
    def forward(self, eh, er, et, ehr_, et_, data, tea_dis, prefix=''):
        
        stu_dis = self.get_stu_dis(eh, er, et, ehr_, et_, data)
    
        stu_dis, _, _ = self.stu_score_preprocess(stu_dis)
        tea_dis, _, _ = self.tea_score_preprocess(tea_dis)        
        
        if type(self.distill_func).__name__ in ['HuberLoss', 'KL_divergency']:
            loss = self.distill_func(stu_dis, tea_dis)
        if type(self.distill_func).__name__ in ['KL_divergency2']:
            mask = only_learn_right_mask(stu_dis, tea_dis)
            loss = self.distill_func(stu_dis, tea_dis, mask)
        
        loss_record = {
            'soft_loss'+prefix: loss.item(),
        }
        
        return loss, loss_record
        
        

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
            'soft_loss'+prefix: loss.item(),
            'fusion_score_loss_record'+prefix: fusion_score_loss.item()
        }
        
        return loss, loss_record


class AlignmentLoss(nn.Module):
    def __init__(self, args, tea_score1_preprocess='constant', tea_score2_preprocess='constant'):
        super(AlignmentLoss, self).__init__()
        self.args = args
        self.tea_score1_preprocess = globals()[tea_score1_preprocess]
        self.tea_score2_preprocess = globals()[tea_score2_preprocess]
    
    def calculate_loss(self,tea_dis1, tea_dis2):
    # 对每个batch的行计算均值和平方的均值
        mean_1 = tea_dis1.mean(dim=1)  # 每一行的均值
        square_mean_1 = (tea_dis1 ** 2).mean(dim=1)  # 每一行平方的均值
        std1 = square_mean_1 - mean_1

        mean_2 = tea_dis2.mean(dim=1)
        square_mean_2 = (tea_dis2 ** 2).mean(dim=1)
        std2 = square_mean_2 - mean_2

        std_diff = (std2-std1)**2
        
        # 对差值取平均数作为最终的loss
        loss = (std_diff.mean()) / (tea_dis1.shape[0])

        return loss
    
    
    def forward(self, tea_dis1, tea_dis2, prefix='Teacher_Aignment'):
        tea_dis1, _, _ = self.tea_score1_preprocess(tea_dis1)
        tea_dis2, _, _ = self.tea_score2_preprocess(tea_dis2)

        loss = self.calculate_loss(tea_dis1, tea_dis2)
        
        loss_record = {
            'soft_loss'+prefix: loss.item(),
        }
        
        return loss, loss_record

