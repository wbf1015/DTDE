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
sys.path.append(new_directory)

"""

================================================归一化函数===================================================

"""
def global_standardize(scores, eps=1e-6):
    mean = scores.mean()
    std = scores.std()
    standardized_tensor = (scores - mean) / (std + eps)
    return standardized_tensor, mean, std

def inverse_global_standardize(standardized_tensor, mean, std, eps=1e-6):
    original_tensor = standardized_tensor * (std + eps) + mean
    return original_tensor

def local_standardize(scores, eps=1e-6):
    scores_mean = scores.mean(dim=-1, keepdim=True)
    scores_sqrtvar = torch.sqrt(scores.var(dim=-1, keepdim=True) + eps)
    scores_norm = (scores - scores_mean) / scores_sqrtvar
    return scores_norm, scores_mean, scores_sqrtvar

def inverse_local_standardize(standardized_tensor, scores_mean, scores_sqrtvar):
    original_tensor = standardized_tensor * scores_sqrtvar + scores_mean
    return original_tensor

def global_minmax(scores, eps=1e-6):
    scores_max = scores.max()
    scores_min = scores.min()
    scores_norm = (scores - scores_min) / (scores_max - scores_min + eps)
    return scores_norm, scores_max, scores_min

def reverse_global(scores_norm, scores_max, scores_min, eps=1e-6):
    scores = scores_norm * (scores_max - scores_min + eps) + scores_min
    return scores

def local_minmax(scores, eps = 1e-6):
    scores_max, _ = scores.max(dim=-1, keepdim=True)  # [batch, 1]
    scores_min, _ = scores.min(dim=-1, keepdim=True)  # [batch, 1]
    scores_norm = (scores - scores_min) / (scores_max - scores_min + eps)  # Min-Max归一化
    return scores_norm, scores_max, scores_min

def reverse_local(scores_norm, scores_max, scores_min, eps=1e-6):
    scores = scores_norm * (scores_max - scores_min + eps) + scores_min
    return scores

def adjust_var(tensor, k):
    mean = tensor.mean(dim=1, keepdim=True)
    variance = tensor.var(dim=1, unbiased=False, keepdim=True)
    adjusted_tensor = mean + (tensor - mean) * torch.sqrt(k)
    
    return adjusted_tensor

def constant(scores):
    return scores, None, None

"""CVPR 2024"""
def logits_normalize(logits):
    mean = logits.mean(dim=-1, keepdim=True)
    std = logits.std(dim=-1, keepdim=True)
    return (logits - mean) / (std + 1e-6), None, None