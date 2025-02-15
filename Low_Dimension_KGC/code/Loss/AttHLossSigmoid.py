import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class AttHLossSigmoid(nn.Module):
    def __init__(self, adv_temperature = None, margin = 6.0, args=None):
        super(AttHLossSigmoid, self).__init__()
        self.args=args
        self.margin = nn.Parameter(torch.Tensor([margin]))
        self.margin.requires_grad = False
        if adv_temperature != None:
            self.adv_temperature = nn.Parameter(torch.Tensor([adv_temperature]))
            self.adv_temperature.requires_grad = False
            self.adv_flag = True
            logging.info(f'Init AttHLossSigmoid with adv_temperature={adv_temperature}')
        else:
            self.adv_flag = False
            logging.info('Init AttHLossSigmoid without adv_temperature')

    def forward(self, p_score, n_score, subsampling_weight=None, sub_margin=False, add_margin=False):
        
        n_score = n_score.reshape(-1, 1)
        p_score = p_score.unsqueeze(-1)
        positive_score = F.logsigmoid(p_score)
        negative_score = F.logsigmoid(-n_score)
        
        positive_loss = -positive_score.mean()
        negative_loss = -negative_score.mean()

        loss = - torch.cat([positive_score, negative_score], dim=0).mean()
        
        loss_record = {
            'hard_positive_sample_loss': positive_loss.item(),
            'hard_negative_sample_loss': negative_loss.item(),
            'hard_loss': loss.item(),
        }
        
        return loss, loss_record

    def predict(self, p_score, n_score):
        score = self.forward(p_score, n_score)
        return score.cpu().data.numpy()