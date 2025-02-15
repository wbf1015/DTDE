import sys
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, y_pred, y_true, reduction='batchmean'):
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

class SCCFLOSS(nn.Module):
    def __init__(self, temperature, args):
        super(SCCFLOSS, self).__init__()
        self.temperature = temperature
        self.args = args
    
    # 非线性部分用根号表示，但是会惩罚不相关的项。
    def similarity1_2(self, ehr, et):
        dot_product = torch.sum(ehr * et, dim=-1)
        norm_product = torch.norm(ehr, p=2, dim=-1) * torch.norm(et, p=2, dim=-1)

        # 非线性部分改成开根号，取绝对值后开根号，并根据dot_product的正负决定最终的符号
        sqrt_term = torch.sqrt(torch.abs(dot_product / norm_product))

        # 根据 dot_product 的正负决定加上还是减去 sqrt_term 项
        exp_term = torch.exp(dot_product / (self.temperature * norm_product))
        adjusted_exp_term = exp_term + torch.sign(dot_product) * torch.exp(sqrt_term / self.temperature)

        sim = adjusted_exp_term

        return sim
    
    # 没有非线性的部分
    def similarity1(self, ehr, et):
        dot_product = torch.sum(ehr * et, dim=-1)
        norm_product = torch.norm(ehr, p=2, dim=-1) * torch.norm(et, p=2, dim=-1)

        sim = torch.exp(dot_product / (self.temperature * norm_product))

        return sim
    
    # 非线性部分使用2来进行
    def similarity2(self, ehr, et):
        dot_product = torch.sum(ehr * et, dim=-1)
        norm_product = torch.norm(ehr, p=2, dim=-1) * torch.norm(et, p=2, dim=-1)

        sim = torch.exp(dot_product / (self.temperature * norm_product)) + torch.exp((dot_product / norm_product) ** 2 / self.temperature)

        return sim
    
    
    def similarity3(self, ehr, et):
        dot_product = torch.sum(ehr * et, dim=-1)
        norm_product = torch.norm(ehr, p=2, dim=-1) * torch.norm(et, p=2, dim=-1)

        sim = torch.exp(dot_product / (self.temperature * norm_product)) + torch.exp((dot_product / norm_product) ** 3 / self.temperature)
        return sim
    
    def SGL_WA(self, ehr, et, tau):
        numerator = torch.exp(torch.tensor(1.0 / tau))
    
        similarities = torch.exp(torch.sum(ehr * et, dim=-1) / tau)
         
        denominator = torch.sum(similarities, dim=1) + numerator  # Shape: [batch]
        
        log_prob = -torch.log(numerator / denominator)
        
        loss = log_prob.mean()
        
        return loss
    
    def forward(self, ehr, et, subsampling_weight):
        similarity = self.similarity3(ehr, et)
        
        # 如果是fully-negative的话
        if torch.all(subsampling_weight == 1.0):
            # cl_loss = self.SGL_WA(ehr, et, self.args.cl_tau)
            # loss = cl_loss 
            # loss_record = {
            #     'cl_loss': cl_loss.item(),
            #     'hard_loss': loss.item(),
            #     }
            # return loss, loss_record
        
            n_score = similarity
            if self.args.negative_adversarial_sampling is True:
                n_score = (F.softmax(n_score * self.args.adversarial_temperature, dim = 1).detach() * n_score).sum(dim = 1)
            else:
                n_score = n_score.mean(dim=-1)
            negative_sample_loss = torch.log(n_score.mean())
            loss = negative_sample_loss
            loss_record = {
            'hard_negative_sample_loss': negative_sample_loss.item(),
            'hard_loss': loss.item(),
            }
            return loss, loss_record
        
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
        
        # print(positive_sample_loss, negative_sample_loss)
        
        loss = (positive_sample_loss + negative_sample_loss)/2
        loss_record = {
            'hard_positive_sample_loss': positive_sample_loss.item(),
            'hard_negative_sample_loss': negative_sample_loss.item(),
            'hard_loss': loss.item(),
        }
        
        return loss, loss_record
    
    def predict(self, ehr, et):
        similarity = self.similarity3(ehr, et)
        
        return similarity
        

class MarginSCCFLoss(nn.Module):
    def __init__(self, temperature, args):
        super(MarginSCCFLoss, self).__init__()
        self.temperature = temperature
        self.args = args
        self.margin = args.gamma
    
    # 非线性部分使用根号下来表示
    def similarity1_2(self, ehr, et):
        dot_product = torch.sum(ehr * et, dim=-1)
        norm_product = torch.norm(ehr, p=2, dim=-1) * torch.norm(et, p=2, dim=-1)

        sim = torch.exp(dot_product / (self.temperature * norm_product)) + torch.exp((dot_product / norm_product) ** 0.5 / self.temperature)

        return sim
    
    # 没有非线性的部分
    def similarity1(self, ehr, et):
        dot_product = torch.sum(ehr * et, dim=-1)
        norm_product = torch.norm(ehr, p=2, dim=-1) * torch.norm(et, p=2, dim=-1)

        sim = torch.exp(dot_product / (self.temperature * norm_product))

        return sim
    
    # 非线性部分使用2来进行
    def similarity2(self, ehr, et):
        dot_product = torch.sum(ehr * et, dim=-1)
        norm_product = torch.norm(ehr, p=2, dim=-1) * torch.norm(et, p=2, dim=-1)

        sim = torch.exp(dot_product / (self.temperature * norm_product)) + torch.exp((dot_product / norm_product) ** 2 / self.temperature)

        return sim
    
    def forward(self, ehr, et, subsampling_weight):
        similarity = self.similarity2(ehr, et)

        p_score, n_score = similarity[:, 0], similarity[:, 1:]
        
        
        if self.args.negative_adversarial_sampling is True:
            p_score = F.logsigmoid(p_score-self.margin)
            n_score = (F.softmax(n_score * self.args.adversarial_temperature, dim = 1).detach() * F.logsigmoid(self.margin-n_score)).sum(dim = 1)
        else:
            p_score = F.logsigmoid(p_score-self.margin)
            n_score = F.logsigmoid(self.margin-n_score).mean(dim = 1)
        
        if self.args.subsampling:
            positive_sample_loss = -((subsampling_weight * p_score).sum() / subsampling_weight.sum())
            negative_sample_loss = -((subsampling_weight * n_score).sum() / subsampling_weight.sum())
        else:
            positive_sample_loss = - (p_score.mean())
            negative_sample_loss = - (n_score.mean())
            
        loss = (positive_sample_loss + negative_sample_loss)/2
        loss_record = {
            'hard_positive_sample_loss': positive_sample_loss.item(),
            'hard_negative_sample_loss': negative_sample_loss.item(),
            'hard_loss': loss.item(),
        }
        
        return loss, loss_record
    
    def predict(self, ehr, et):
        similarity = self.similarity(ehr, et)
        
        return similarity


class Combine_hr(nn.Module):
    def __init__(self, entity_dim=512, relation_dim=512, hidden_dim=32, layer_mul=2):
        super(Combine_hr, self).__init__()
        self.entity_dim=entity_dim
        self.relation_dim=relation_dim
        self.hidden_dim=hidden_dim
        self.layer_mul=layer_mul
        self.MLP = nn.Sequential(
            nn.Linear((entity_dim+relation_dim), (entity_dim+relation_dim)*self.layer_mul),
            # nn.ReLU(),
            nn.Linear((entity_dim+relation_dim)*self.layer_mul, hidden_dim)
        )
    
    def forward(self, eh, er):
        # Concatenate eh and er along the last dimension
        combined = torch.cat((eh, er), dim=2)

        # Pass through the MLP
        output = self.MLP(combined)
        
        return output


class Combine_hr2(nn.Module):
    def __init__(self, entity_dim=512, relation_dim=512, hidden_dim=32, layer_mul=2):
        super(Combine_hr2, self).__init__()
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
        # Concatenate eh and er along the last dimension
        combined = torch.cat((eh, er), dim=2)  # Shape: [batch, 1, entity_dim + relation_dim]

        # Reshape for BatchNorm: [batch_size * seq_len, hidden_dim]
        batch_size, seq_len, _ = combined.size()
        combined = combined.view(batch_size * seq_len, -1)

        # Pass through the MLP
        output = self.MLP(combined)

        # Reshape output back to [batch_size, seq_len, hidden_dim]
        output = output.view(batch_size, seq_len, self.hidden_dim)
        
        return output


class Tail_Transform(nn.Module):
    def __init__(self, input_dim=64, output_dim=32, layer_mul=2):
        super(Tail_Transform, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_mul = layer_mul
        self.MLP = nn.Sequential(
            nn.Linear((input_dim), (input_dim)*self.layer_mul),
            # nn.ReLU(),
            nn.Linear((input_dim)*self.layer_mul, output_dim)
        )
        
    
    def forward(self, tail):        
        # Pass through the MLP
        output = self.MLP(tail)
        
        return output


class Tail_Transform2(nn.Module):
    def __init__(self, input_dim=64, output_dim=32, layer_mul=2):
        super(Tail_Transform2, self).__init__()
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


'''
从原始的模型通过转换得到的低维模型，并不是直接的表示，需要进行头实体和关系的融合
如果关系的shape不合适同样也要融合
'''
class SCCF_Decoder(nn.Module):
    def __init__(self, args):
        super(SCCF_Decoder, self).__init__()
        self.args = args
        self.target_dim = args.target_dim
        self.entity_dim = args.target_dim*args.entity_mul
        self.relation_dim = args.target_dim * args.relation_mul
        self.SCCFLOSS = SCCFLOSS(self.args.temprature, args)
        self.Combine_hr = Combine_hr(self.entity_dim, self.relation_dim, self.target_dim, layer_mul=2)
        self.tail_transform = Tail_Transform(self.entity_dim, self.target_dim)
        logging.info('Init SCCF_Decoder successfully')
    
    def forward(self, eh, er, et, subsampling_weight):
        ehr = self.Combine_hr(eh, er)
        if ehr.shape[-1]!=et.shape[-1]:
            et = self.tail_transform(et)

        loss, loss_record = self.SCCFLOSS(ehr, et, subsampling_weight)
        
        return loss, loss_record

    def predict(self, eh, er, et):
        ehr = self.Combine_hr(eh, er)
        if ehr.shape[-1]!=et.shape[-1]:
            et = self.tail_transform(et)
        score = self.SCCFLOSS.predict(ehr, et)
        
        return score


'''
在某些情况下，我们可以直接在语义的提取和低维嵌入转换阶段就得到一个大小固定的嵌入
这个时候就不需要转换了
'''
class SCCF_Decoder2(nn.Module):
    def __init__(self, args):
        super(SCCF_Decoder2, self).__init__()
        self.args = args
        self.SCCFLOSS = SCCFLOSS(self.args.temprature, args)
        logging.info('Init SCCF_Decoder successfully')
    
    def forward(self, ehr, et, subsampling_weight):
        assert ehr.shape[-1] == et.shape[-1]
        loss, loss_record = self.SCCFLOSS(ehr, et, subsampling_weight)
        return loss, loss_record

    def predict(self, ehr, et):
        assert ehr.shape[-1] == et.shape[-1]
        score = self.SCCFLOSS.predict(ehr, et)
        return score



class SCCF_Decoder_2KGE(nn.Module):
    def __init__(self, args):
        super(SCCF_Decoder_2KGE, self).__init__()
        self.args = args
        self.target_dim = args.target_dim
        self.entity_dim = args.target_dim * args.entity_mul
        self.relation_dim = args.target_dim * args.relation_mul
        self.SCCFLOSS = SCCFLOSS(self.args.temprature, args)
        self.Combine_hr = Combine_hr2(self.entity_dim, self.relation_dim, self.target_dim, layer_mul=2)
        self.tail_transform = Tail_Transform2(self.entity_dim, self.target_dim)
        
        self.hyperbolic_gate = torch.tensor(0.3)
        self.euclid_gate = torch.tensor(0.4)
        self.SCCF_LN = nn.LayerNorm(normalized_shape=self.args.negative_sample_size+1)
        self.PT1_LN = nn.LayerNorm(normalized_shape=self.args.negative_sample_size+1)
        self.PT2_LN = nn.LayerNorm(normalized_shape=self.args.negative_sample_size+1)
        
        self.scaling = nn.Parameter(torch.ones(1))
        self.Huber_Loss = HuberLoss()
        
        logging.info('Init SCCF_Decoder_2KGE successfully')
    
    
    def cal_soft_loss(self, SCCF_score, PT1_score, PT2_score, subsampling_weight):        
        #前者相对于后者的偏移
        def minmax_norm(scores):
            scores_max, _ = scores.max(dim=1, keepdim=True)  # [batch, 1]
            scores_min, _ = scores.min(dim=1, keepdim=True)  # [batch, 1]
            
            # 防止分母为0的情况，添加一个很小的epsilon
            eps = 1e-8
            scores_norm = (scores - scores_min) / (scores_max - scores_min + eps)  # Min-Max归一化
            
            return scores_norm
        
        def zscore_norm(scores, eps=1e-6):
            scores_mean = scores.mean(dim=1, keepdim=True)  # [batch, 1]
            scores_sqrtvar = torch.sqrt(scores.var(dim=1, keepdim=True) + eps)  # [batch, 1]
            scores_norm = (scores - scores_mean) / scores_sqrtvar  # Z-Score标准化
            return scores_norm
        
        # SCCF_score, PT1_score, PT2_score = minmax_norm(SCCF_score), minmax_norm(PT1_score), minmax_norm(PT2_score)
        # SCCF_score, PT1_score, PT2_score = zscore_norm(SCCF_score), zscore_norm(PT1_score), zscore_norm(PT2_score)

        divergence_PT1 = F.kl_div(SCCF_score.softmax(-1).log(), PT1_score.softmax(-1), reduction='batchmean')
        divergence_PT2 = F.kl_div(SCCF_score.softmax(-1).log(), PT2_score.softmax(-1), reduction='batchmean')
        
        # divergence_PT1 = F.kl_div(SCCF_score.softmax(-1).log(), PT1_score.softmax(-1).detach(), reduction='batchmean')
        # divergence_PT2 = F.kl_div(SCCF_score.softmax(-1).log(), PT2_score.softmax(-1).detach(), reduction='batchmean')        

        loss = (divergence_PT1 + divergence_PT2)/2
        
        loss_record = {
            'SCCF_PT1_divergence_loss': divergence_PT1.item(),
            'SCCF_PT2_divergence_loss': divergence_PT2.item(),
            'soft_loss': loss.item(),
        }
        
        return loss, loss_record
    
    def cal_soft_loss2(self, SCCF_score, PT1_score, PT2_score, subsampling_weight):
        def cal_soft_loss(SCCF_score, PT_score, div_PT_score, gate):
            mean_PT = PT_score.mean(dim=0, keepdim=True)
            variance_PT = (PT_score - mean_PT) ** 2
            batch_max_values = variance_PT.max(dim=1, keepdim=True).values
            adjusted_max_values = batch_max_values * gate  # [batch_size, 1]
            relu_softmax_variance_PT = F.relu(variance_PT - adjusted_max_values)
            
            # zero_count = (relu_softmax_variance_PT == 0).sum().item()
            # print(f"每个batch包含张量中包含 {(zero_count)//self.args.batch_size} 个 0")
            mask = (relu_softmax_variance_PT > 0).float()
            # mask = 1-mask
            # zero_count = (mask != 0).sum().item()
            # print(f"每个batch包含张量中包含 {(zero_count)//self.args.batch_size} 个 1")
            # print(f"总共有{zero_count}个1")
            # print(mask)
            
            masked_input = relu_softmax_variance_PT + (mask - 1) * 1e10
            masked_softmax_output = F.softmax(masked_input, dim=1)
            # masked_softmax_output = softmax_output * mask
            
            masked_divergence_PT = div_PT_score * masked_softmax_output
            soft_loss = masked_divergence_PT.sum(dim=-1).mean()
            
            return soft_loss
        
        # epsilon = 1e-10
        # divergence_PT1 = F.kl_div((SCCF_score.softmax(-1)+epsilon).log(), PT1_score.softmax(-1)+epsilon, reduction='none')
        # divergence_PT2 = F.kl_div((SCCF_score.softmax(-1)+epsilon).log(), PT2_score.softmax(-1)+epsilon, reduction='none')
        divergence_PT1 = F.kl_div(SCCF_score.softmax(-1).log(), PT1_score.softmax(-1), reduction='none')
        divergence_PT2 = F.kl_div(SCCF_score.softmax(-1).log(), PT2_score.softmax(-1), reduction='none')
        
        divergence_PT1, divergence_PT2 = torch.abs(divergence_PT1), torch.abs(divergence_PT2)
        
        divergence_PT1 = cal_soft_loss(SCCF_score, PT1_score, divergence_PT1, self.hyperbolic_gate)
        divergence_PT2 = cal_soft_loss(SCCF_score, PT2_score, divergence_PT2, self.euclid_gate)
        
        loss = (divergence_PT1 + divergence_PT2)/2
        
        loss_record = {
            'SCCF_PT1_divergence_loss': divergence_PT1.item(),
            'SCCF_PT2_divergence_loss': divergence_PT2.item(),
            'soft_loss': loss.item(),
        }
        
        return loss, loss_record
    
    def cal_soft_loss3(self, SCCF_score, PT1_score, PT2_score, subsampling_weight):
        def compute_weights(scores: torch.Tensor) -> torch.Tensor:
            sorted_indices = torch.argsort(scores, dim=1, descending=True)  # Shape: [batch, neg_sampling_size]
            first_element_ranks = (sorted_indices == 0).nonzero(as_tuple=True)[1] + 1  # Shape: [batch]
            weights = 1.0 / torch.log2(first_element_ranks.float() + 1)
            weights = weights.view(-1, 1)
            return weights
        
        PT1_weight = compute_weights(PT1_score)
        PT2_weight = compute_weights(PT2_score)
        
        weights_softmax = F.softmax(torch.cat((PT1_weight, PT2_weight), dim=1), dim=1)
        PT1_weight, PT2_weight = torch.split(weights_softmax, 1, dim=1)
        
        divergence_PT1 = F.kl_div(SCCF_score.softmax(-1).log(), PT1_score.softmax(-1), reduction='none')
        divergence_PT2 = F.kl_div(SCCF_score.softmax(-1).log(), PT2_score.softmax(-1), reduction='none')
        
        divergence_PT1 = (divergence_PT1 * PT1_weight).mean()
        divergence_PT2 = (divergence_PT2 * PT2_weight).mean()
        
        loss = (divergence_PT1 + divergence_PT2)/2
        
        loss_record = {
            'SCCF_PT1_divergence_loss': divergence_PT1.item(),
            'SCCF_PT2_divergence_loss': divergence_PT2.item(),
            'soft_loss': loss.item(),
        }
        
        return loss, loss_record
        
    def cal_soft_loss4(self, SCCF_score, PT1_score, PT2_score, subsampling_weight):
        def standardize(tensor, eps=1e-6):
            mean = tensor.mean()
            std = tensor.std()
            standardized_tensor = (tensor - mean) / (std + eps)
            return standardized_tensor, mean, std
        
        def minmax(tensor, eps=1e-6):
            scores_max = tensor.max()
            scores_min = tensor.min()
            scores_norm = (tensor - scores_min) / (scores_max - scores_min + eps)
            return scores_norm
        
        def map_distribution(standard_tensor, target_mean, target_std):
            return standard_tensor * target_std + target_mean
        
        def minmax_norm(scores):
            scores_max, _ = scores.max(dim=1, keepdim=True)  # [batch, 1]
            scores_min, _ = scores.min(dim=1, keepdim=True)  # [batch, 1]
            
            # 防止分母为0的情况，添加一个很小的epsilon
            eps = 1e-8
            scores_norm = (scores - scores_min) / (scores_max - scores_min + eps)  # Min-Max归一化
            
            return scores_norm
        
        def zscore_norm(scores, eps=1e-6):
            scores_mean = scores.mean(dim=1, keepdim=True)  # [batch, 1]
            scores_sqrtvar = torch.sqrt(scores.var(dim=1, keepdim=True) + eps)  # [batch, 1]
            scores_norm = (scores - scores_mean) / scores_sqrtvar  # Z-Score标准化
            return scores_norm
        
        PT1_score  = minmax(PT1_score)
        PT2_score  = minmax(PT2_score)
        PT_score = PT1_score + PT2_score
        
        divergence_PT = F.kl_div(SCCF_score.softmax(-1).log(), PT_score.softmax(-1), reduction='batchmean')
        
        loss = divergence_PT
        
        loss_record = {
            'SCCF_PT_divergence_loss': divergence_PT.item(),
            'soft_loss': loss.item(),
        }
        
        return loss, loss_record
    
    def cal_soft_loss5(self, SCCF_score, PT1_score, PT2_score, subsampling_weight):
        def minmax_norm(scores, eps=1e-8):
            scores_max, _ = scores.max(dim=1, keepdim=True)  # [batch, 1]
            scores_min, _ = scores.min(dim=1, keepdim=True)  # [batch, 1]
            
            # 防止分母为0的情况，添加一个很小的epsilon
            scores_norm = (scores - scores_min) / (scores_max - scores_min + eps)  # Min-Max归一化
            
            return scores_norm
        
        def zscore_norm(scores, eps=1e-8):
            scores_mean = scores.mean(dim=1, keepdim=True)  # [batch, 1]
            scores_sqrtvar = torch.sqrt(scores.var(dim=1, keepdim=True) + eps)  # [batch, 1]
            scores_norm = (scores - scores_mean) / scores_sqrtvar  # Z-Score标准化
            return scores_norm
        
        # SCCF_score, PT1_score, PT2_score  = minmax_norm(SCCF_score), minmax_norm(PT1_score), minmax_norm(PT2_score) 
        SCCF_score, PT1_score, PT2_score  = zscore_norm(SCCF_score), zscore_norm(PT1_score), zscore_norm(PT2_score) 
        HL_PT1 = self.Huber_Loss(SCCF_score, PT1_score, reduction='mean')
        HL_PT2 = self.Huber_Loss(SCCF_score, PT2_score, reduction='mean')
        
        loss = (HL_PT1 + HL_PT2)/2
        
        loss_record = {
            'HL_PT1': HL_PT1.item(),
            'HL_PT2': HL_PT2.item(),
            'soft_loss': loss.item(),
        }
        
        return loss, loss_record
        
    
    def DmutDE_cal_soft_loss(self, SCCF_score, PT1_score, PT2_score, subsampling_weight):
        def minmax_norm(scores):
           scores_max, _ = scores.max(-1, keepdim=True)
           scores_min, _ = scores.min(-1, keepdim=True)
           scores_norm = (scores - scores_min) / (scores_max - scores_min)
           return scores_norm[:, 0:1], scores_norm[:, 1:]
        
        def zscore_norm(scores, eps=1e-6):
            scores_mean = scores.mean(-1, keepdim=True)
            scores_sqrtvar = torch.sqrt(scores.var(-1, keepdim=True) + eps)
            scores_norm = (scores - scores_mean) / scores_sqrtvar
            return scores_norm[:, 0:1], scores_norm[:, 1:]
        
        def neg_dist_learning(scaling, neg_score_s, neg_score_t):
            neg_score_s_ = F.softmax(scaling * neg_score_s, dim=1).clone().detach() * neg_score_s
            neg_score_t_ = F.softmax(scaling * neg_score_t, dim=1).clone().detach() * neg_score_t
            return self.kl_loss(F.log_softmax(neg_score_s_, dim=1), F.softmax(neg_score_t_.detach(), dim=1))
        
        pos_PT1_score, neg_PT1_score = zscore_norm(PT1_score)
        pos_PT2_score, neg_PT2_score = zscore_norm(PT2_score)
        
        pos_indicats = (pos_PT1_score > pos_PT2_score).float().detach()
        neg_indicats = (neg_PT1_score < neg_PT2_score).float().detach()
        
        pos_mixscore_t = pos_indicats * pos_PT1_score + (1. - pos_indicats) * pos_PT2_score
        neg_mixscore_t = neg_indicats * pos_PT1_score + (1. - neg_indicats) * pos_PT2_score
        
        PT_score =  torch.cat([pos_mixscore_t, neg_mixscore_t], dim=1)
        
        distill_loss = F.kl_div(F.log_softmax(SCCF_score, dim=-1), (F.softmax(PT_score, dim=-1)).detach())
        loss = distill_loss
        
        loss_record = {
            'soft_loss': loss.item(),
        }
        
        return loss, loss_record

        
        
    
    def forward(self, eh, er, et, PT1_score, PT2_score, data=None, subsampling_weight=None):
        ehr = self.Combine_hr(eh, er)
        if et.shape[-1] != ehr.shape[-1]:
            et = self.tail_transform(et)

        SCCF_score = self.SCCFLOSS.predict(ehr, et)
        
        hard_loss, loss_record = self.SCCFLOSS(ehr, et, subsampling_weight)
        soft_loss, loss_record2 = self.cal_soft_loss(SCCF_score, PT1_score, PT2_score, subsampling_weight)
        # soft_loss, loss_record2 = self.cal_soft_loss2(SCCF_score, PT1_score, PT2_score, subsampling_weight)
        # soft_loss, loss_record2 = self.cal_soft_loss3(SCCF_score, PT1_score, PT2_score, subsampling_weight)
        # soft_loss, loss_record2 = self.cal_soft_loss4(SCCF_score, PT1_score, PT2_score, subsampling_weight)
        # soft_loss, loss_record2 = self.cal_soft_loss5(SCCF_score, PT1_score, PT2_score, subsampling_weight)
        # soft_loss, loss_record2 = self.DmutDE_cal_soft_loss(SCCF_score, PT1_score, PT2_score, subsampling_weight)
        
        loss = hard_loss + self.args.kdloss_weight * soft_loss
                
        loss_record.update(loss_record2)
        loss_record.update({'LOSS':loss.item()})
        
        return loss, loss_record

    def predict(self, eh, er, et):
        ehr = self.Combine_hr(eh, er)
        if et.shape[-1] != ehr.shape[-1]:
            et = self.tail_transform(et)
        score = self.SCCFLOSS.predict(ehr, et)
        
        return score




class MarginSCCF_Decoder(nn.Module):
    def __init__(self, args):
        super(MarginSCCF_Decoder, self).__init__()
        self.args = args
        self.target_dim = args.target_dim
        self.entity_dim = args.target_dim * args.entity_mul
        self.relation_dim = args.target_dim * args.relation_mul
        self.SCCFLOSS = MarginSCCFLoss(self.args.temprature, args)
        self.Combine_hr = Combine_hr(self.entity_dim, self.relation_dim, self.target_dim, layer_mul=2)
        self.tail_transform = Tail_Transform(self.entity_dim, self.target_dim)
        logging.info('Init SCCF_Decoder successfully')
    
    def forward(self, eh, er, et, subsampling_weight):
        ehr = self.Combine_hr(eh, er)
        if ehr.shape[-1]!=et.shape[-1]:
            et = self.tail_transform(et)
        loss, loss_record = self.SCCFLOSS(ehr, et, subsampling_weight)
        
        return loss, loss_record

    def predict(self, eh, er, et):
        ehr = self.Combine_hr(eh, er)
        if ehr.shape[-1]!=et.shape[-1]:
            et = self.tail_transform(et)
        score = self.SCCFLOSS.predict(ehr, et)
        
        return score





if __name__ == "__main__":
    loss = SCCFLOSS(0.1)
    ehr = torch.rand((1024, 1, 512))
    et = torch.rand((1024, 257, 512))
    sim = loss.similarity(ehr, et)
    p_score = sim[:,0]
    print(sim.shape)
    print(p_score.shape)