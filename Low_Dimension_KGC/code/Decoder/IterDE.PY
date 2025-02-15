import sys
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np



class Decoder_IterDE(nn.Module):
    def __init__(self, args):
        super(Decoder_IterDE, self).__init__()
        self.args = args
        self.gamma = self.args.pos_gamma
        self.KGE = RotatE_Reverse(self.gamma)
    
    
    def forward(self, eh, er, et, PT1_score, PT2_score, data=None, subsampling_weight=None, mode='1posNneg'):
        stu_score = self.KGE(eh, er, et, self.args)
        
        
        return loss, loss_record
    
    def predict(self, eh, er, et):
        
        return stu_score
    
    def predict2(self, eh, er, et):
        
        return stu_score

    def predict3(self, eh, er, et):
        
        return stu_score



class RotatE_Reverse(nn.Module):
    def __init__(self, margin=None, embedding_range=11.0, embedding_dim=512):
        super(RotatE_Reverse, self).__init__()
        # RotatE中的成员变量记录的是教师模型的信息，真正学生的参数在args中保存
        self.margin = margin
        self.origin_margin = embedding_range - 2.0
        self.embedding_range = embedding_range
        self.embedding_dim = embedding_dim
        
        self.trainable_margin = nn.Parameter(torch.tensor([[self.origin_margin]]))
        
        logging.info(f'Init RotatE_Reverse with embedding_range={self.embedding_range}, embedding_dim={self.embedding_dim}, margin={self.margin}')
    
    def forward(self, head, relation, tail, args):
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]
        
        if head.shape[-1]>500:
            phase_relation = relation/(((self.embedding_range)/self.embedding_dim)/pi)
        else:
            embedding_range, embedding_dim = 2.0+args.gamma, args.target_dim
            phase_relation = relation/(((embedding_range)/embedding_dim)/pi)
        
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)


        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        re_score = re_score - re_tail
        im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = score.sum(dim = 2)
        
        if head.shape[-1]>500:
            score = self.origin_margin - score
        else:
            if self.margin is not None:
                score = self.margin - score
            else:
                score = score
        
        return score


class KDLoss(nn.Module):
    def __init__(self, adv_temperature = None, margin = 6.0, delta=1.0, args=None, teacher_margin=9.0, need_sub=True):
        super(KDLoss, self).__init__()
        self.args = args
        self.delta = delta
        self.margin = nn.Parameter(torch.Tensor([margin]))
        self.margin.requires_grad = False
        self.teacher_margin = teacher_margin
        self.need_sub = need_sub
        
        if adv_temperature != None:
            self.adv_temperature = nn.Parameter(torch.Tensor([adv_temperature]))
            self.adv_temperature.requires_grad = False
            self.adv_flag = True
            logging.info(f'Init KDLoss with adv_temperature={adv_temperature}')
        else:
            self.adv_flag = False
            logging.info('Init KDLoss without adv_temperature')

    '''
    reference:https://juejin.cn/s/pytorch%20huber%20loss%20example
    '''
    def huber_loss(self, t_score, s_score, subsampling_weight=None):
        residual = torch.abs(t_score - s_score)
        condition = (residual < self.delta).float()
        loss = condition * 0.5 * residual**2 + (1 - condition) * (self.delta * residual - 0.5 * self.delta**2)
        loss = self.margin - loss
               
        p_loss, n_loss = loss[:, 0], loss[:, 1:]
        
        if self.adv_flag:
            n_loss = (F.softmax(n_loss * self.adv_temperature, dim = 1).detach() 
                              * F.logsigmoid(n_loss)).sum(dim = 1)
        else:
            n_loss = F.logsigmoid(n_loss).mean(dim = 1)
        p_loss = F.logsigmoid(p_loss)
        
        if self.args.subsampling==False:
            p_loss = - p_loss.mean()
            n_loss = - n_loss.mean()
        else:
            p_loss = - (subsampling_weight * p_loss).sum()/subsampling_weight.sum()
            n_loss = - (subsampling_weight * n_loss).sum()/subsampling_weight.sum()
        
        loss = (p_loss + n_loss)/2
        
        return loss, p_loss, n_loss
        
    
    def forward(self, t_score, s_score, subsampling_weight=None):
        if self.need_sub:
            t_score, s_score = self.teacher_margin-t_score, self.student_margin-s_score
        
        loss, p_loss, n_loss = self.huber_loss(t_score, s_score, subsampling_weight=subsampling_weight)
        
        loss_record = {
            'positive_huber_loss': p_loss.item(),
            'negative_huber_loss': n_loss.item(),
            'huber_loss': loss.item(),
        }
        
        return loss, loss_record


class SigmoidLossOrigin(nn.Module):
    def __init__(self, adv_temperature = None, margin = 6.0, args=None):
        super(SigmoidLossOrigin, self).__init__()
        self.args=args
        self.margin = nn.Parameter(torch.Tensor([margin]))
        self.margin.requires_grad = False
        if adv_temperature != None:
            self.adv_temperature = nn.Parameter(torch.Tensor([adv_temperature]))
            self.adv_temperature.requires_grad = False
            self.adv_flag = True
            logging.info(f'Init SigmoidLossOrigin with adv_temperature={adv_temperature}')
        else:
            self.adv_flag = False
            logging.info('Init SigmoidLossOrigin without adv_temperature')

    def forward(self, p_score, n_score, subsampling_weight=None, sub_margin=False, add_margin=False):
        if sub_margin:
            p_score, n_score = self.margin-p_score, self.margin-n_score
        if add_margin:
            p_score, n_score = self.margin+p_score, self.margin+n_score
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
                
        # 到这里就是1*1的了
        loss = (positive_sample_loss + negative_sample_loss)/2
        
        loss_record = {
            'hard_positive_sample_loss': positive_sample_loss.item(),
            'hard_negative_sample_loss': negative_sample_loss.item(),
            'hard_loss': loss.item(),
        }
        
        return loss, loss_record

    def predict(self, p_score, n_score):
        score = self.forward(p_score, n_score)
        return score.cpu().data.numpy()