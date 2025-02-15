import sys
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

"""

=========================================================嵌入融合变换模块============================================

"""     
    
class Combine_hr(nn.Module):
    def __init__(self, entity_dim=512, relation_dim=512, hidden_dim=32, layer_mul=2):
        super(Combine_hr, self).__init__()
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.hidden_dim = hidden_dim
        self.layer_mul = layer_mul
        
        # Define the MLP with BatchNorm
        self.MLP = nn.Sequential(
            nn.Linear((entity_dim + relation_dim), (entity_dim + relation_dim) * self.layer_mul),
            nn.LeakyReLU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear((entity_dim + relation_dim) * self.layer_mul, hidden_dim),
            nn.BatchNorm1d(hidden_dim)  # Batch normalization on hidden_dim dimension
        )
    
    def forward(self, eh, er):
        if eh.shape[1]==er.shape[1]:
            combined = torch.cat((eh, er), dim=2)  # Shape: [batch, 1, entity_dim + relation_dim]
        else:                                      # 为predict_augment所保留的代码
            er = er.expand(-1, eh.shape[1], -1)
            combined = torch.cat((eh, er), dim=2)
        batch_size, seq_len, _ = combined.size()
        combined = combined.view(batch_size * seq_len, -1)
        output = self.MLP(combined)
        output = output.view(batch_size, seq_len, self.hidden_dim)
        return output


class Tail_Transform(nn.Module):
    def __init__(self, input_dim=64, output_dim=32, layer_mul=2):
        super(Tail_Transform, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_mul = layer_mul
        
        # Define the MLP with BatchNorm
        self.MLP = nn.Sequential(
            nn.Linear(input_dim, input_dim * self.layer_mul),
            nn.LeakyReLU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear(input_dim * self.layer_mul, output_dim),
            nn.BatchNorm1d(output_dim)  # Batch normalization on output_dim dimension
        )
    
    def forward(self, tail):
        # If tail has more than 2 dimensions, reshape for BatchNorm
        if tail.dim() > 2:
            batch_size, seq_len, _ = tail.size()
            tail = tail.view(batch_size * seq_len, -1)

            # Pass through the MLP
            output = self.MLP(tail)

            # Reshape output back to original sequence format
            output = output.view(batch_size, seq_len, self.output_dim)
        else:
            # Directly pass through MLP if tail is already 2D
            output = self.MLP(tail)
        
        return output

class Condition_Tail_Transform(nn.Module):
    def __init__(self, input_dim=64, output_dim=32, layer_mul=2):
        super(Condition_Tail_Transform, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_mul = layer_mul
        
        # Define the MLP with BatchNorm
        self.MLP = nn.Sequential(
            nn.Linear(input_dim*2, input_dim * self.layer_mul),
            nn.LeakyReLU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear(input_dim * self.layer_mul, output_dim),
            nn.BatchNorm1d(output_dim)  # Batch normalization on output_dim dimension
        )
    
    def forward(self, query, tail):
        if query.shape[1] < tail.shape[1]:
            query = query.expand(-1, tail.shape[1], -1)
            condition_tail = torch.cat((query, tail), dim=-1)
        else:
            tail = tail.expand(-1, query.shape[1], -1)
            condition_tail = torch.cat((query, tail), dim=-1)
        
        # If tail has more than 2 dimensions, reshape for BatchNorm
        if condition_tail.dim() > 2:
            batch_size, seq_len, _ = condition_tail.size()
            condition_tail = condition_tail.view(batch_size * seq_len, -1)

            # Pass through the MLP
            output = self.MLP(condition_tail)

            # Reshape output back to original sequence format
            output = output.view(batch_size, seq_len, self.output_dim)
        else:
            # Directly pass through MLP if tail is already 2D
            output = self.MLP(condition_tail)
        
        return output

class BN(nn.Module):
    def __init__(self, input_dim=32):
        super(BN, self).__init__()
        self.input_dim=input_dim
        self.BatchNorm = nn.BatchNorm1d(input_dim)
    
    def forward(self, t):
        batch_size, seq_len, _ = t.size()
        t = t.view(batch_size * seq_len, -1)
        t = self.BatchNorm(t)
        t = t.view(batch_size, seq_len, -1)
        return t

class Constant(nn.Module):
    def __init__(self, ):
        super(Constant, self).__init__()
    
    def forward(self, t):
        return t
