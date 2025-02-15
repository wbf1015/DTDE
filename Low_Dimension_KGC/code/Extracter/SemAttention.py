import sys
import os
import logging
import torch
import math
import torch.nn as nn
import numpy as np

CODEPATH = os.path.abspath(os.path.dirname(__file__))
CODEPATH = CODEPATH.rsplit('/', 1)[0]
sys.path.append(CODEPATH)

from Transformers.PoswiseFeedForwardNet import *
from Transformers.ScaleDotAttention import *
from Transformers.SelfAttention import *

class LGSemAttention(nn.Module):
    def __init__(self, input_dim, output_dim, subspace, n_heads, d_ff=1, LN=False):
        super(LGSemAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.subspace = subspace
        self.n_heads = n_heads
        self.d_ff = d_ff                      
        self.LN = LN        
        
        self.subspace_sem = nn.ModuleList([
            nn.Linear(input_dim // subspace, output_dim)
            for _ in range(subspace)
        ])
        self.global_sem = nn.Linear(subspace*output_dim+input_dim, output_dim)
        
        self.sem_fusion= SelfAttention4(output_dim, output_dim, output_dim, n_heads, residual='V')
        self.forgetgate = nn.Linear(output_dim+input_dim, output_dim)
        if LN is False:
            self.fc = PoswiseFeedForwardNet2(output_dim, d_ff=d_ff)
        else:
            self.fc = PoswiseFeedForwardNet(output_dim, d_ff=d_ff)
        self.sigmoid = nn.Sigmoid()
        
        logging.info(f'Init LGSemAttention with input_dim={self.input_dim}, output_dim={self.output_dim}, subspace={self.subspace}, n_heads={self.n_heads}, d_ff={self.d_ff}, sem_fusion={self.sem_fusion.__class__.__name__}, forgetgate=nn.Linear, fc={self.fc.__class__.__name__}, sigmoid=nn.Sigmoid')

    def get_sem(self, enc_inputs):
        batch_size, neg_sampling, dim = enc_inputs.shape
        
        # 分割enc_inputs
        chunks = torch.chunk(enc_inputs, self.subspace, dim=-1)
        
        # 通过subspace_sem处理每一份
        processed_chunks = [layer(chunk).unsqueeze(-2) for chunk, layer in zip(chunks, self.subspace_sem)]

        # 将结果合并为所需形状
        combined = torch.cat(processed_chunks, dim=-2)
        
        # 保存副本并重塑
        reshaped_combined = combined.view(batch_size, neg_sampling, -1)
        
        # 与原始输入拼接并通过global_sem处理
        global_input = torch.cat((reshaped_combined, enc_inputs), dim=-1)
        global_output = self.global_sem(global_input)
        
        
        Semantic = torch.cat((combined, global_output.unsqueeze(2)), dim=2)
        
        return Semantic
    
    
    def forward(self, enc_inputs, forget):
        semantic = self.get_sem(enc_inputs)
        outputs = self.sem_fusion(semantic, semantic, semantic)
        outputs = outputs.mean(dim=-2)
        outputs = self.fc(outputs)
        
        forget_ = self.forgetgate(torch.cat((outputs, forget), dim=-1)) # 可以探索一下是使用过fc的outputs还是不过fc的outputs
        forget_ = outputs * self.sigmoid(forget_) # 加还是*都可以试试 StageAttention1/2
        # # forget_ = outputs + forget_
        # forget_ = outputs + outputs * self.sigmoid(forget_)
        
        return outputs, forget_


class SemUpdate(nn.Module):
    def __init__(self, sem_dim, embedding_dim, LN=False):
        super(SemUpdate, self).__init__()
        self.sem_dim = sem_dim
        self.embedding_dim = embedding_dim
        self.LN = LN        
        
        self.reset_weight = nn.Linear(sem_dim+embedding_dim, sem_dim)
        self.update_weight = nn.Linear(sem_dim+embedding_dim, sem_dim)
        self.reset_transfer = nn.Linear(sem_dim, sem_dim)
        self.update = nn.Linear(sem_dim+embedding_dim, sem_dim)
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
        logging.info(f'Init SemUpdate with sem_dim={self.sem_dim}, embedding_dim={self.embedding_dim}, LN={self.LN}')
        
    def forward(self, sem, origin_embedding):
        reset = self.sigmoid(self.reset_weight(torch.cat((sem, origin_embedding), dim=-1)))
        update = self.sigmoid(self.update_weight(torch.cat((sem, origin_embedding), dim=-1)))
        
        h = self.tanh(self.update(torch.cat((origin_embedding, self.reset_transfer(sem)*reset), dim=-1)))
        outputs = (1-update) * sem + update * h

        if self.LN:
            return nn.LayerNorm(self.sem_dim).cuda()(outputs) 
        else:
            return outputs


class SemAttention(nn.Module):
    def __init__(self, args, Is_Entity=True):
        super(SemAttention, self).__init__()
        self.args = args
        # 因为是Encoder-Decoder结构才会直接到semantic-embedding就停止了，所以语义的大小就由target_dim决定
        if Is_Entity:
            self.layer1 = LGSemAttention(args.input_dim*args.entity_mul, args.target_dim*args.entity_mul, args.token1, args.head1, d_ff=args.t_dff)
            self.layer2 = SemUpdate(args.target_dim*args.entity_mul, args.input_dim*args.entity_mul)
        else:
            self.layer1 = LGSemAttention(args.input_dim*args.relation_mul, args.target_dim*args.relation_mul, args.token1, args.head1, d_ff=args.t_dff)
            self.layer2 = SemUpdate(args.target_dim*args.relation_mul, args.input_dim*args.relation_mul)
        
    def forward(self, inputs):
        outputs = inputs
        
        outputs, _ = self.layer1(outputs, outputs)
        outputs = self.layer2(outputs, inputs)
        
        return outputs