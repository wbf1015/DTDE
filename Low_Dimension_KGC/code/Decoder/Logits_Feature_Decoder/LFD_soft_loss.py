import sys
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

"""

========================================知识蒸馏损失函数（软损失函数）===========================================

"""

class Margin_HuberLoss(nn.Module):
    def __init__(self, args, delta=1.0):
        super(Margin_HuberLoss, self).__init__()
        self.delta = delta
        self.args = args
        self.HuberLoss_gamma = 9.0
    
    def forward(self, s_score, t_score):
        residual = torch.abs(t_score - s_score)
        condition = (residual < self.delta).float()
        loss = condition * 0.5 * residual**2 + (1 - condition) * (self.delta * residual - 0.5 * self.delta**2)
        
        loss = self.HuberLoss_gamma - loss
               
        p_loss, n_loss = loss[:, 0], loss[:, 1:]
        
        n_loss = F.logsigmoid(n_loss).mean(dim = 1)
        p_loss = F.logsigmoid(p_loss)
        
        p_loss = - p_loss.mean()
        n_loss = - n_loss.mean()

        # loss = (p_loss + n_loss)/2
        loss = p_loss*(1/(self.args.negative_sample_size+1)) + n_loss*(self.args.negative_sample_size/(self.args.negative_sample_size+1))
        
        return loss


class HuberLoss(nn.Module):
    def __init__(self, args, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta
        self.args = args
    
    def forward(self, y_pred, y_true, reduction='mean'):
        residual = torch.abs(y_true - y_pred)
        mask = residual < self.delta
        loss = torch.where(mask, 0.5 * residual ** 2, self.delta * residual - 0.5 * self.delta ** 2)
        
        if reduction=='batchmean':
            loss = loss.sum()/y_pred.shape[0]
        elif reduction=='sum':
            loss = loss.sum()
        elif reduction=='mean':
            loss = loss.mean()
        else:
            loss = loss
        
        return loss


class KL_divergency(nn.Module):
    def __init__(self, args, temprature=None):
        super(KL_divergency, self).__init__()
        self.args = args
        self.softmax = nn.Softmax(dim=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        if temprature is None:
            self.temprature_TS = self.args.temprature_ts
        else:
            self.temprature_TS = temprature

    def forward_(self, student_p, teacher_p, reduction='batchmean'):
        loss = F.kl_div(torch.log(student_p), teacher_p, reduction=reduction) * self.temprature_TS * self.temprature_TS
        return loss
    
    def cal_p(self, student_dis, teacher_dis):
        student_p = self.softmax(student_dis/self.temprature_TS)
        teacher_p = self.softmax(teacher_dis/self.temprature_TS)
        
        return student_p, teacher_p
    
    def forward(self, student_dis, teacher_dis, reduction='batchmean'):
        student_p, teacher_p = self.cal_p(student_dis, teacher_dis)
        loss = self.forward_(student_p, teacher_p, reduction)
        
        return loss


"""
CVPR2024 Logit_Standard KD
"""
class DistillKL_Logit_Standard(nn.Module):
    """Logit Standardization in Knowledge Distillation"""
    def __init__(self, args):
        super(DistillKL_Logit_Standard, self).__init__()
        self.args = args
        self.softmax = nn.Softmax(dim=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.temperature = self.args.temprature_ts  # 确保有这个属性

    def normalize(self, logits):
        # 标准化logits
        mean = logits.mean(dim=-1, keepdim=True)
        std = logits.std(dim=-1, keepdim=True)
        return (logits - mean) / (std + 1e-6)

    def forward(self, student_logits, teacher_logits, reduction='batchmean'):
        T = self.temperature
        student_p = self.logsoftmax(self.normalize(student_logits) / T)
        teacher_p = self.softmax(self.normalize(teacher_logits) / T)
        
        # 计算知识蒸馏损失
        loss = F.kl_div(student_p, teacher_p, reduction=reduction) * (T * T)
        return loss

"""
用来处理输入直接是概率的情况
"""
class KL_divergencyv2(nn.Module):
    def __init__(self, args, temprature=None):
        super(KL_divergencyv2, self).__init__()
        self.args = args
        self.softmax = nn.Softmax(dim=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        if temprature is None:
            self.temprature_TS = self.args.temprature_ts
        else:
            self.temprature_TS = temprature

    def forward_(self, student_p, teacher_p, reduction='batchmean'):
        loss = F.kl_div(torch.log(student_p), teacher_p, reduction=reduction) * self.temprature_TS * self.temprature_TS
        return loss
    
    def forward(self, student_logits, teacher_prob, reduction='batchmean'):
        T = self.temprature_TS
        student_prob = self.softmax(student_logits / T)
        loss = self.forward_(student_prob, teacher_prob, reduction)
        
        return loss