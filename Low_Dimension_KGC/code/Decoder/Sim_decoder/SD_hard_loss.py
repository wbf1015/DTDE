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

from SD_similarity import *

sys.path.remove(new_directory)
'''

=====================================================为1posnneg的硬损失计算===========================================

'''
class SCCF_loss(nn.Module):
    def __init__(self, args):
        super(SCCF_loss, self).__init__()
        self.args = args
    
    def forward(self, similarity, subsampling_weight):
        p_score, n_score = similarity[:, 0], similarity[:, 1:]
        
        if self.args.negative_adversarial_sampling is True:
            log_p_score = torch.log(p_score)
            n_score = (F.softmax(n_score * self.args.adversarial_temperature, dim = 1).detach() * n_score).sum(dim = 1)
        else:
            log_p_score = torch.log(p_score)
            n_score = n_score.mean(dim=-1)
        
        if self.args.subsampling:
            positive_sample_loss = -((subsampling_weight * log_p_score).sum() / subsampling_weight.sum())
            negative_sample_loss = torch.log((subsampling_weight * n_score).sum() / subsampling_weight.sum())
        else:
            positive_sample_loss = ((-1) * log_p_score.mean())
            negative_sample_loss = torch.log(n_score.mean())
        
        loss = (positive_sample_loss + negative_sample_loss)/2
        loss_record = {
            'hard_positive_sample_loss': positive_sample_loss.item(),
            'hard_negative_sample_loss': negative_sample_loss.item(),
            'hard_loss': loss.item(),
        }
        
        return loss, loss_record

class SCCF_loss2(nn.Module):
    def __init__(self, args):
        super(SCCF_loss2, self).__init__()
        self.args = args
    
    def forward(self, similarity, subsampling_weight):
        p_score, n_score = similarity[:, 0], similarity[:, 1:]
        n_pos, n_neg = 1, similarity.shape[-1]-1
        
        if self.args.negative_adversarial_sampling is True:
            log_p_score = torch.log(p_score)
            n_score = (F.softmax(n_score * self.args.adversarial_temperature, dim = 1).detach() * n_score).sum(dim = 1)
        else:
            log_p_score = torch.log(p_score)
            n_score = n_score.mean(dim=-1)
        
        if self.args.subsampling:
            positive_sample_loss = -((subsampling_weight * log_p_score).sum() / subsampling_weight.sum())
            negative_sample_loss = torch.log((subsampling_weight * n_score).sum() / subsampling_weight.sum())
        else:
            positive_sample_loss = ((-1) * log_p_score.mean())
            negative_sample_loss = torch.log(n_score.mean())
        
        loss = (n_pos/(n_pos+n_neg)) * positive_sample_loss + (n_neg/(n_pos+n_neg)) * negative_sample_loss
        
        loss_record = {
            'hard_positive_sample_loss': positive_sample_loss.item(),
            'hard_negative_sample_loss': negative_sample_loss.item(),
            'hard_loss': loss.item(),
        }
        
        return loss, loss_record


class SigmoidLoss(nn.Module):
    def __init__(self, args):
        super(SigmoidLoss, self).__init__()
        self.args = args
        self.pos_margin = nn.Parameter(torch.Tensor([self.args.pos_gamma]))
        self.neg_margin = nn.Parameter(torch.Tensor([self.args.neg_gamma]))
        self.pos_margin.requires_grad = False
        self.neg_margin.requires_grad = False
        if self.args.negative_adversarial_sampling:
            self.adv_temperature = nn.Parameter(torch.Tensor([self.args.adversarial_temperature]))
            self.adv_temperature.requires_grad = False
            self.adv_flag = True
        else:
            self.adv_flag = False
    
    def forward(self, similarity, subsampling_weight=None, small_better=False, big_better=False, sub=False, add=False):
        p_score, n_score = similarity[:, 0], similarity[:, 1:]
        if small_better:
            p_score, n_score = self.pos_margin-p_score, self.neg_margin-n_score
        if big_better:
            p_score, n_score = p_score-self.pos_margin, n_score-self.neg_margin
        if sub:
            p_score, n_score = self.pos_margin-p_score, self.neg_margin-n_score
        if add: # big better也可以走这个
            p_score, n_score = self.pos_margin+p_score, self.neg_margin+n_score
        if self.adv_flag:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(n_score * self.adv_temperature, dim = 1).detach()
                            * F.logsigmoid(-n_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-n_score).mean(dim = 1)
            
        positive_score = F.logsigmoid(p_score)
        
        if self.args.subsampling == False:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()
        
        loss = (positive_sample_loss + negative_sample_loss)/2
        loss_record = {
            'hard_positive_sample_loss': positive_sample_loss.item(),
            'hard_negative_sample_loss': negative_sample_loss.item(),
            'hard_loss': loss.item(),
        }
        return loss, loss_record


class SigmoidLoss2(nn.Module):
    def __init__(self, args, pos_margin=None, neg_margin=None):
        super(SigmoidLoss2, self).__init__()
        self.args = args
        if (pos_margin is None) and (neg_margin is None):
            self.pos_margin = nn.Parameter(torch.Tensor([self.args.pos_gamma]))
            self.neg_margin = nn.Parameter(torch.Tensor([self.args.neg_gamma]))
            self.pos_margin.requires_grad = False
            self.neg_margin.requires_grad = False
            self.prefix = ''
        else:
            self.pos_margin = nn.Parameter(torch.Tensor([pos_margin]))
            self.neg_margin = nn.Parameter(torch.Tensor([neg_margin]))
            self.pos_margin.requires_grad = False
            self.neg_margin.requires_grad = False
            self.prefix = 'FT_teacher_'

        if self.args.negative_adversarial_sampling:
            self.adv_temperature = nn.Parameter(torch.Tensor([self.args.adversarial_temperature]))
            self.adv_temperature.requires_grad = False
            self.adv_flag = True
        else:
            self.adv_flag = False
    
    def forward(self, similarity, subsampling_weight=None, small_better=False, big_better=False, sub=False, add=False):
        p_score, n_score = similarity[:, 0], similarity[:, 1:]
        n_pos, n_neg = 1, similarity.shape[-1]-1
        if small_better:
            p_score, n_score = self.pos_margin-p_score, self.neg_margin-n_score
        if big_better:
            p_score, n_score = p_score-self.pos_margin, n_score-self.neg_margin
        if sub:
            p_score, n_score = self.pos_margin-p_score, self.neg_margin-n_score
        if add: # big better也可以走这个
            p_score, n_score = self.pos_margin+p_score, self.neg_margin+n_score
        if self.adv_flag:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(n_score * self.adv_temperature, dim = 1).detach()
                            * F.logsigmoid(-n_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-n_score).mean(dim = 1)
            
        positive_score = F.logsigmoid(p_score)
        
        if self.args.subsampling == False:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()
        
        if self.prefix=='':
            loss = (n_pos/(n_pos+n_neg)) * positive_sample_loss + (n_neg/(n_pos+n_neg)) * negative_sample_loss
        else:
            loss = (positive_sample_loss+negative_sample_loss)/2
        
        loss_record = {
            self.prefix + 'hard_positive_sample_loss': positive_sample_loss.item(),
            self.prefix + 'hard_negative_sample_loss': negative_sample_loss.item(),
            self.prefix + 'hard_loss': loss.item(),
        }
        return loss, loss_record


class SigmoidLoss_KnowledgeInjection(nn.Module):
    def __init__(self, args, pos_margin=None):
        super(SigmoidLoss_KnowledgeInjection, self).__init__()
        self.args = args
        if pos_margin is None:
            self.pos_margin = nn.Parameter(torch.Tensor([self.args.pos_gamma]))
        else:
            self.pos_margin = nn.Parameter(torch.Tensor([pos_margin]))
        self.pos_margin.requires_grad = False
        if self.args.negative_adversarial_sampling:
            self.adv_temperature = nn.Parameter(torch.Tensor([self.args.adversarial_temperature]))
            self.adv_temperature.requires_grad = False
            self.adv_flag = True
        else:
            self.adv_flag = False
    
    def forward(self, similarity, subsampling_weight=None, small_better=False, big_better=False, sub=False, add=False):
        p_score = similarity
        if small_better:
            p_score = self.pos_margin-p_score
        if big_better:
            p_score = p_score-self.pos_margin
        if sub:
            p_score = self.pos_margin-p_score
        if add: # big better也可以走这个
            p_score = self.pos_margin+p_score
            
        positive_score = F.logsigmoid(p_score)
        
        if self.args.subsampling == False:
            positive_sample_loss = - positive_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
        
        loss = positive_sample_loss
        loss_record = {
            'knowledge_injection_loss': positive_sample_loss.item(),
        }
        return loss, loss_record


class BCELoss(nn.Module):
    def __init__(self, args):
        super(BCELoss, self).__init__()
        self.args = args
        self.bceloss = torch.nn.BCELoss()
        self.sigmoid = torch.nn.Sigmoid()
        self.label_smooth = 0.1
    
    def forward(self, similarity, subsampling_weight):
        target = torch.zeros_like(similarity)
        target[:, 1:] = self.label_smooth/(similarity.shape[-1]-1)
        target[:, 0] = 1-self.label_smooth
        similarity = self.sigmoid(similarity)
        loss = self.bceloss(similarity, target)
        loss_record = {
            'hard_loss': loss.item(),
        }
        return loss, loss_record




'''

=====================================================为all true的硬损失计算===========================================

'''

class SigmoidLoss_AT(nn.Module):
    def __init__(self, args):
        super(SigmoidLoss_AT, self).__init__()
        self.args = args
        self.npositive  = args.positive_sample_size
        self.margin = nn.Parameter(torch.Tensor([self.args.gammaTrue]))
        self.margin.requires_grad = False
        if self.args.negative_adversarial_sampling:
            self.adv_temperature = nn.Parameter(torch.Tensor([self.args.adversarial_temperature]))
            self.adv_temperature.requires_grad = False
            self.adv_flag = True
        else:
            self.adv_flag = False
        self.true_triples = self.get_true_triples()
        self.cal_sim = cal_similarity(args=self.args, temperature=self.args.temprature)
    
    def get_true_triples(self, ):
        train_true_triples = {}
        
        train_data_path = self.args.data_path + '/train'
        relation_num = self.args.nrelation//2
        
        with open(train_data_path) as fin:
            for line in fin:
                h, r, t = line.strip().split('\t')
                h, r, t = int(h), int(r), int(t)
                if (h,r) not in train_true_triples:
                    train_true_triples[(h,r)] = []
                train_true_triples[(h,r)].append(t)
                
                if(t, r+relation_num) not in train_true_triples:
                    train_true_triples[(t,r+relation_num)] = []
                train_true_triples[(t,r+relation_num)].append(h)
        
        return train_true_triples
    
    def generate_mask(self, data):
        """
        生成mask矩阵。

        :param data: 包含 (positive_sample, negative_sample) 的元组。
        :return: 大小为(batch, npositive)的布尔型tensor mask，设备与 positive_sample 一致。
        """
        positive_sample, _ = data
        # 假设 positive_sample 是张量
        batch_size = positive_sample.shape[0]

        # 提取头实体和关系
        head_relation_pairs = [(int(head), int(relation)) for head, relation, _ in positive_sample]

        # 生成每个 (head, relation) 对应的正样本数量
        true_counts = [
            min(len(self.true_triples[head, relation]), self.npositive)
            for head, relation in head_relation_pairs
        ]
        # 构建 mask，设备与 positive_sample 一致
        mask = torch.zeros((batch_size, self.npositive+1), dtype=torch.bool, device=positive_sample.device)

        for i, true_count in enumerate(true_counts):
            mask[i, :true_count+1] = True

        return mask
    
    def get_batch_mean(self, et, mask):
        mask = mask.unsqueeze(-1)
        masked_et = et * mask
        sum_masked_et = masked_et.sum(dim=1)
        valid_count = mask.sum(dim=1)
        average_et = sum_masked_et / valid_count
        return average_et
    
    def forward(self, embedding, similarity, data, subsampling_weight=None, small_better=False, big_better=False, sub=False, add=False):
        if small_better or sub:
            score = self.margin - similarity
        if big_better or add:
            score = similarity - self.margin
        
        mask = self.generate_mask(data)
        
        ehr, et = embedding
        batchmean_et = self.get_batch_mean(et, mask)
        mean_vec_score = self.cal_sim.SCCF_similarity3(ehr, batchmean_et.unsqueeze(1))
        
        score = similarity
        mean_vec_score =  F.logsigmoid(mean_vec_score)
        
        score = F.logsigmoid(score)
        
        score = score*mask
        score = score.mean(dim=-1)
        
        if self.args.subsampling == False:
            mean_vec_loss =  - mean_vec_score.mean()
            loss = - score.mean()
        else:
            mean_vec_loss = - (subsampling_weight * mean_vec_loss).sum()/subsampling_weight.sum()
            loss = - (subsampling_weight * score).sum()/subsampling_weight.sum()
        
        loss_record = {
            'mean_vec_loss' : mean_vec_loss.item(),
            'hard_loss' : loss.item()
        }

        loss = mean_vec_loss + loss
        # loss = loss
        return loss, loss_record