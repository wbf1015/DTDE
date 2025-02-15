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
    
from LFD_norm import *

"""

=========================================教师模型得分融合 AND 损失函数融合=================================

"""
def get_PT2_loss(soft_loss1, soft_loss2):
    return soft_loss2

def loss_half_fusion(soft_loss1, soft_loss2):
    loss = (soft_loss1 + soft_loss2) / 2
    return loss

def scores_add_fusion(PT1_score, PT2_score):
    PT_score = PT1_score + PT2_score
    return PT_score

# 输入的数据是归一化之后的，返回的也是归一化的数据。
def scores_dmutde_fusion(PT1_score, PT2_score):
    # 分解输入分数
    pos_PT1_score, neg_PT1_score = PT1_score[:, 0:1], PT1_score[:, 1:]  # 保证维度
    pos_PT2_score, neg_PT2_score = PT2_score[:, 0:1], PT2_score[:, 1:]  # 保证维度

    # 计算指示器
    pos_indicats = (pos_PT1_score > pos_PT2_score).to(torch.float32).detach()
    neg_indicats = (neg_PT1_score < neg_PT2_score).to(torch.float32).detach()

    # 计算混合分数
    pos_mixscore_t = pos_indicats * pos_PT1_score + (1.0 - pos_indicats) * pos_PT2_score
    neg_mixscore_t = neg_indicats * neg_PT1_score + (1.0 - neg_indicats) * neg_PT2_score

    # 拼接正负分数
    PT_score = torch.cat([pos_mixscore_t, neg_mixscore_t], dim=1)

    return PT_score

"""
把第二个教师分布映射到第一个教师分布（根据均值-方差）
"""
def scores_fusionv1(PT1_score, PT2_score, weight=None):
    mean_PT1, std_PT1 = PT1_score.mean(dim=1, keepdim=True), PT1_score.std(dim=1, keepdim=True) # 计算PT1_score的均值和标准差
    mean_PT2, std_PT2 = PT2_score.mean(dim=1, keepdim=True), PT2_score.std(dim=1, keepdim=True) # 计算PT2_score的均值和标准差
    
    PT2_score_normalized = (PT2_score - mean_PT2) / std_PT2 # 标准化PT2_score
    PT2_score_mapped = PT2_score_normalized * std_PT1 + mean_PT1 # 将标准化后的PT2_score映射到PT1_score的分布
    
    if weight is None:
        fusion_score = (PT1_score + PT2_score_mapped)/2
    else:
        weight_PT1 = weight[:, :, 0]  # 提取PT1_score的权重并增加一个维度以匹配分数的形状
        weight_PT2 = weight[:, :, 1]  # 提取PT2_score_mapped的权重并增加一个维度
        fusion_score = weight_PT1 * PT1_score + weight_PT2 * PT2_score_mapped  # 计算加权平均分数
    
    return fusion_score 

"""
把第一个教师分布映射到第二个教师分布（根据均值-方差）
"""
def scores_fusionv2(PT1_score, PT2_score, weight=None):
    # 计算PT1_score和PT2_score的均值和标准差
    mean_PT1, std_PT1 = PT1_score.mean(dim=1, keepdim=True), PT1_score.std(dim=1, keepdim=True)
    mean_PT2, std_PT2 = PT2_score.mean(dim=1, keepdim=True), PT2_score.std(dim=1, keepdim=True)
    

    PT1_score_normalized = (PT1_score - mean_PT1) / std_PT1
    PT1_score_mapped = PT1_score_normalized * std_PT2 + mean_PT2 # 将PT1_score映射到PT2_score的分布
    
    if weight is None:
        fusion_score = (PT1_score_mapped + PT2_score) / 2
    else:
        weight_PT1 = weight[:, :, 0]  # 提取PT1_score的权重并增加一个维度以匹配分数的形状
        weight_PT2 = weight[:, :, 1] # 提取PT2_score_mapped的权重并增加一个维度
        fusion_score = weight_PT1 * PT1_score_mapped + weight_PT2 * PT2_score  # 计算加权平均分数
    
    return fusion_score

"""
把每一个教师分数分布的方差降下来
"""
def scores_fusionv3(PT1_score, PT2_score, weight=None):
    
    scale = 1.5
    PT1_score, PT2_score = PT1_score/scale, PT2_score/scale
    
    fusion_score = scores_fusionv2(PT1_score, PT2_score, weight)
    
    return fusion_score

"""
处理可能的”错误值“
"""
def scores_fusionv4(PT1_score, PT2_score, weight=None):
    
    first_elements = PT1_score[:, 0:1]  # 获取每一行的第一个元素shape [batch, 1]
    filled_first_elements = first_elements.expand_as(PT1_score) # 创建一个与PT1_score相同形状的tensor，每一行都填充对应的第一个元素
    PT1_score_clamped = torch.min(PT1_score, filled_first_elements) # 使用torch.min将原始scores中大于第一个元素的值替换为第一个元素的值

    first_elements_PT2 = PT2_score[:, 0:1] # 对PT2_score执行相同的操作
    filled_first_elements_PT2 = first_elements_PT2.expand_as(PT2_score)
    PT2_score_clamped = torch.min(PT2_score, filled_first_elements_PT2)

    fusion_score = scores_fusionv3(PT1_score_clamped, PT2_score_clamped, weight=weight)
    
    # 返回处理后的scores
    return fusion_score

"""
直接对softmax之后的概率值做融合, 需要搭配KL_divergencyv2使用
"""
def scores_fusionv5(PT1_score, PT2_score, weight, temprature_TS=1.0):
    PT1_prob = F.softmax(PT1_score/temprature_TS, dim=-1)
    PT2_prob = F.softmax(PT2_score/temprature_TS, dim=-1)
    
    weight_PT1 = weight[:, :, 0]  # 提取PT1_score的权重并增加一个维度以匹配分数的形状
    weight_PT2 = weight[:, :, 1]  # 提取PT2_score_mapped的权重并增加一个维度
    PT_prob = PT1_prob * weight_PT1  + PT2_prob * weight_PT2
    
    return PT_prob


"""
都归一化，再映射回去
"""
def scores_fusionv6(PT1_score, PT2_score, weight=None):
    # 计算PT1_score和PT2_score的均值和标准差
    mean_PT1, std_PT1 = PT1_score.mean(dim=1, keepdim=True), PT1_score.std(dim=1, keepdim=True)
    mean_PT2, std_PT2 = PT2_score.mean(dim=1, keepdim=True), PT2_score.std(dim=1, keepdim=True)

    PT1_score_normalized = (PT1_score - mean_PT1) / std_PT1
    PT2_score_normalized = (PT2_score - mean_PT2) / std_PT2
    
    if weight is None:
        fusion_score = (PT1_score_normalized + PT2_score_normalized) / 2
    else:
        weight_PT1 = weight[:, :, 0]  # 提取PT1_score的权重并增加一个维度以匹配分数的形状
        weight_PT2 = weight[:, :, 1] # 提取PT2_score_mapped的权重并增加一个维度
        fusion_score = weight_PT1 * PT1_score_normalized + weight_PT2 * PT2_score_normalized  # 计算加权平均分数
        
    mean = mean_PT1 + mean_PT2
    std = torch.sqrt(std_PT1**2 + std_PT2**2)
    
    fusion_score = fusion_score * std + mean
    
    return fusion_score


"""
不对分布做任何调整
"""
def scores_fusionv7(PT1_score, PT2_score, weight=None):
    
    if weight is None:
        fusion_score = (PT1_score + PT2_score) / 2
    else:
        weight_PT1 = weight[:, :, 0]  # 提取PT1_score的权重并增加一个维度以匹配分数的形状
        weight_PT2 = weight[:, :, 1] # 提取PT2_score_mapped的权重并增加一个维度
        fusion_score = weight_PT1 * PT1_score + weight_PT2 * PT2_score  # 计算加权平均分数
    
    return fusion_score