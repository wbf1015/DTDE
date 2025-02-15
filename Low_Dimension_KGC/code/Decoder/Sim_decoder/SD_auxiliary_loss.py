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
    
from SD_norm import *
from SD_hard_loss import *
from SD_soft_loss import *
from SD_similarity import *

"""

=========================================================辅助软损失计算===================================

"""
class DmutDE_neg_soft_loss(nn.Module):
    def __init__(self, args):
        super(DmutDE_neg_soft_loss, self).__init__()
        self.args = args
        self.soft_loss = KL_divergency(self.args)
    
    def forward(self, student_scores, teacher_scores):
        neg_score_s, neg_score_t = student_scores[:, 1:], teacher_scores[:, 1:]
        neg_score_s_ = F.softmax(self.scaling * neg_score_s, dim=1).clone().detach() * neg_score_s
        neg_score_t_ = F.softmax(self.scaling * neg_score_t, dim=1).clone().detach() * neg_score_t
        
        return self.soft_loss(neg_score_s_, neg_score_t_)


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        # 定义两个可训练参数 k 和 b
        self.k = nn.Parameter(torch.randn(1))  # 初始化为随机值
        self.b = nn.Parameter(torch.randn(1))  # 初始化为随机值
        self.solid = nn.Parameter(torch.tensor(0.6), requires_grad=False)

    def forward(self, x):
        x = x.unsqueeze(1)
        # 模型的前向传播计算 y = kx + b
        return self.k * x + self.b + self.solid

"""
Weight Map for Score Fusion, 细粒度的权重学习，通过qauery和tail一起指定权重
"""
class weight_learner(nn.Module):
    def __init__(self, args):
        super(weight_learner, self).__init__()
        self.args=args
        self.entity_dim = args.target_dim * args.entity_mul
        self.relation_dim = args.target_dim * args.relation_mul
        self.layer_mul = 2
        self.MLP = nn.Sequential(
            nn.Linear((self.entity_dim + self.entity_dim), (self.entity_dim + self.entity_dim) * self.layer_mul),
            nn.LeakyReLU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear((self.entity_dim + self.entity_dim) * self.layer_mul, 2)
        )

    def forward(self, query, tail, data=None):
        query_expanded = query.expand(-1, tail.size(1), -1)
        combined = torch.cat((query_expanded, tail), dim=2) 
        
        x = self.MLP(combined)
        weights = F.softmax(x, dim=2)
        
        return weights

"""
Weight Map for Score Fusion, 粗粒度的权重学习，通过qauery指定权重
"""
class weight_learnerv2(nn.Module):
    def __init__(self, args):
        super(weight_learnerv2, self).__init__()
        self.args=args
        self.entity_dim = args.target_dim * args.entity_mul
        self.relation_dim = args.target_dim * args.relation_mul
        self.layer_mul = 2
        
        self.MLP = nn.Sequential(
            nn.Linear((self.entity_dim), (self.entity_dim) * self.layer_mul),
            nn.LeakyReLU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear((self.entity_dim) * self.layer_mul, 2)
        )

    def forward(self, query, tail, data=None):
        x = self.MLP(query) # x.shape = [batch,1,2]
        x = x.expand(-1, tail.size(1), -1)
        weights = F.softmax(x, dim=2)
        
        return weights

"""
Weight Map for Score Fusion, 细粒度的权重学习，通过qauery和tail一起指定权重
并且为每个entity和relation重新学习了一个embedding专门用于学习权重。
"""
class weight_learnerv3(nn.Module):
    def __init__(self, args):
        super(weight_learnerv3, self).__init__()
        self.args=args
        self.data_type = torch.double if self.args.data_type == 'double' else torch.float
        self.entity_dim = args.target_dim * args.entity_mul
        self.relation_dim = args.target_dim * args.relation_mul
        self.layer_mul = 2
        
        self.entity_embedding = nn.Parameter(torch.empty(self.args.nentity, 32, dtype=self.data_type), requires_grad=True)
        self.relation_embedding = nn.Parameter(torch.empty(self.args.nrelation, 32, dtype=self.data_type), requires_grad=True)
        
        nn.init.xavier_uniform_(self.relation_embedding)
        nn.init.xavier_uniform_(self.entity_embedding)
        
        self.PT1_embedding = nn.Parameter(torch.randn(32))
        self.PT2_embedding = nn.Parameter(torch.randn(32))
        self.fusion_embedding = nn.Parameter(torch.randn(32))
        
        self.MLP = nn.Sequential(
            nn.Linear((4 * self.entity_dim  + self.relation_dim), (4 * self.entity_dim  + self.relation_dim) * self.layer_mul),
            nn.LeakyReLU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear((4 * self.entity_dim  + self.relation_dim) * self.layer_mul, 2)
        )
        
        self.MLP_ = nn.Sequential(
            nn.Linear((3 * self.entity_dim), (3 * self.entity_dim) * self.layer_mul),
            nn.LeakyReLU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear((3 * self.entity_dim) * self.layer_mul, 3)
        )
        
        
    def forward_(self, sample):
        head, tail = self.EntityEmbeddingExtract(self.entity_embedding, sample)
        relation = self.RelationEmbeddingExtract(self.relation_embedding, sample)
        return head, relation, tail
    
    def EntityEmbeddingExtract(self, entity_embedding, sample):
        
        if entity_embedding is None:
            return None, None
        
        positive, negative = sample
        batch_size, negative_sample_size = negative.size(0), negative.size(1)
        
        neg_tail = torch.index_select(
            entity_embedding, 
            dim=0, 
            index=negative.view(-1)
        ).view(batch_size, negative_sample_size, -1)
        
        pos_tail = torch.index_select(
            entity_embedding, 
            dim=0, 
            index=positive[:, 2]
        ).unsqueeze(1)
        
        tail = torch.cat((pos_tail, neg_tail), dim=1)
        
        head = torch.index_select(
            entity_embedding, 
            dim=0, 
            index=positive[:, 0]
        ).unsqueeze(1)
            
        return head, tail

    def RelationEmbeddingExtract(self, relation_embedding, sample):
        
        if relation_embedding is None:
            return None
        
        positive, negative = sample
        
        relation = torch.index_select(
                relation_embedding, 
                dim=0, 
                index=positive[:, 1]
            ).unsqueeze(1)
        
        return relation
    
    def forward(self, query, tail, data):
        head_, relation_, tail_ = self.forward_(data)
        query_expand = torch.cat((query, head_, relation_), dim=-1)
        query_expand = query_expand.expand(-1, tail.size(1), -1)
        
        tail_expand = torch.cat((tail, tail_), dim=-1)
        combined = torch.cat((query_expand, tail_expand), dim=-1)
        x = self.MLP(combined)
        weights = F.softmax(x, dim=2)
        
        return weights
    
    def forward2(self, ):
        obj_embedding = torch.cat((self.PT1_embedding, self.PT2_embedding, self.fusion_embedding), dim=0)
        x = self.MLP_(obj_embedding)
        x = F.softmax(x, dim=0)  # 对平均后的结果应用softmax
        return x


"""
知识注入损失
"""
class Knowledge_Injection(nn.Module):
    def __init__(self, args):
        super(Knowledge_Injection, self).__init__()
        self.args = args
        self.threshold = 5
        self.ki_cl_tau = 1.0
        self.cal_sim = cal_similarity(args=self.args, temperature=self.args.temprature)
        self.hard_loss = SigmoidLoss_KnowledgeInjection(args=self.args, pos_margin=4.0)
    
    def get_mask(self, ehr, et, PT1_score, PT2_score):
        # 创建mask，初始值为0
        mask1 = torch.zeros_like(PT1_score)
        mask2 = torch.zeros_like(PT2_score)
        
        # 获取每一行的第一列元素
        first_col1 = PT1_score[:, 0].unsqueeze(1)
        first_col2 = PT2_score[:, 0].unsqueeze(1)
        
        # 计算差距
        diff1 = PT1_score - first_col1
        diff2 = PT2_score - first_col2
        
        # 设置mask的值
        mask1[diff1 > 0] = 1
        mask1[(diff1 < 0) & (diff1 >= -first_col1 * self.threshold)] = 0.5
        
        mask2[diff2 > 0] = 1
        mask2[(diff2 < 0) & (diff2 >= -first_col2 * self.threshold)] = 0.5
        
        # 创建条件mask
        # condition_mask = (mask1 == 1) & (mask2 == 1) | (mask1 == 0.5) & (mask2 == 0.5) | (mask1 == 1) & (mask2 == 0.5) | (mask1 == 0.5) & (mask2 == 1)
        # condition_mask = (mask1 == 0.5) & (mask2 == 0.5) | (mask1 == 1) & (mask2 == 0.5) | (mask1 == 0.5) & (mask2 == 1) # 层次对比学习，threshold=8-10
        condition_mask = (mask1 == 0.5) & (mask2 == 0.5) | (mask1 == 1) & (mask2 == 0.5) | (mask1 == 0.5) & (mask2 == 1) | (mask1 == 0.5) | (mask2 == 0.5) # 层次对比学习 threshol=4-6
        condition_mask[:, 0] = 0
        
        # true_count = condition_mask.sum().item()
        # print('true_count=',true_count)
        return condition_mask
    
    def get_mask2(self, ehr, et, PT1_score, PT2_score):
        # 创建mask，初始值为0
        mask1 = torch.zeros_like(PT1_score)
        mask2 = torch.zeros_like(PT2_score)
        
        PT1_score, _, _ = local_standardize(PT1_score)
        PT2_score, _, _ = local_standardize(PT2_score)
        
        # 获取每一行的第一列元素
        first_col1 = PT1_score[:, 0].unsqueeze(1)
        first_col2 = PT2_score[:, 0].unsqueeze(1)
        
        # 计算差距
        diff1 = PT1_score - first_col1
        diff2 = PT2_score - first_col2
        
        mask1[diff1 > 0] = 1
        mask1[(diff1 < 0) & (diff1 >= -first_col1 * self.threshold)] = 0.5
        
        mask2[diff2 > 0] = 1
        mask2[(diff2 < 0) & (diff2 >= -first_col2 * self.threshold)] = 0.5
        
        # condition_mask = (mask1 == 1) & (mask2 == 1) | (mask1 == 0.5) & (mask2 == 0.5) | (mask1 == 1) & (mask2 == 0.5) | (mask1 == 0.5) & (mask2 == 1)
        condition_mask = (mask1 == 0.5) & (mask2 == 0.5) | (mask1 == 1) & (mask2 == 0.5) | (mask1 == 0.5) & (mask2 == 1) | (mask1 == 0.5) | (mask2 == 0.5)
        condition_mask[:, 0] = 0
        
        # true_count = condition_mask.sum().item()
        # print('true_count=',true_count)
        
        return condition_mask
        
    
    
    """
    用hard-loss来注入知识
    """
    def get_loss(self, ehr, et, PT1_score, PT2_score, mask):
        # 选择符合条件的tail元素，并确保保留batch索引
        selected_tails = et[mask]
        batch_indices = torch.where(mask.any(dim=1))[0]

        # 对应batch的第一个tail,注意并不是找所有行，只是找那一行里有True的那些行
        first_tails = et[batch_indices, 0, :]

        # 为了确保selected_tails和first_tails形状一致，我们需要对first_tails进行重复扩展
        # 找出每个batch中符合条件的tail的数量
        tail_counts = mask.sum(dim=1)[batch_indices]

        # 重复每个batch的第一个tail，使其与selected_tails的数量一致
        repeated_first_tails = first_tails.repeat_interleave(tail_counts, dim=0)

        # 选择对应的query
        selected_queries = ehr[batch_indices, 0, :]

        # 重复每个batch的query，使其与selected_tails的数量一致
        repeated_queries = selected_queries.repeat_interleave(tail_counts, dim=0)

        
        # print(selected_tails.shape, repeated_first_tails.shape, repeated_queries.shape)
        
        # 逐位相加
        modified_tails = 1 * selected_tails + repeated_first_tails
        
        # 拼接成N*dim的二维tensor
        final_querys = repeated_queries
        final_tails = modified_tails
        
        similarity = self.cal_sim.SCCF_similarity3(final_querys, final_tails)
        similarity = similarity.squeeze()
        
        loss, loss_record = self.hard_loss(similarity, big_better=True)
        
        return loss, loss_record
    
    """
    用传统的对比学习来注入知识
    """
    def get_loss2_(self, ehr, et, PT1_score, PT2_score, mask):
        
        first_embeddings = et[:, 0, :].unsqueeze(1)
        et = et + torch.where(mask.unsqueeze(-1), first_embeddings, torch.zeros_like(et))
        
        # 计算相似度分数
        similarity = F.cosine_similarity(ehr, et, dim=-1)
        similarity = similarity/self.ki_cl_tau
        similarity = torch.exp(similarity)
        
        # 计算每一行的正例，包括mask为True的位置和每行的第一个元素
        positive_mask = mask.clone()
        positive_mask[:, 0] = 1  # 将每行的第一个元素设置为正例

        # 对每一行计算对比学习的损失
        # 分子是mask或第一个元素为True的位置的相似度加和
        numerator = torch.sum(similarity * positive_mask, dim=1)
        # 分母是每一行的相似度加和
        denominator = torch.sum(similarity, dim=1)

        # 避免除零错误，添加一个小的epsilon
        epsilon = 1e-8
        loss_per_sample = -torch.log(numerator / (denominator + epsilon))

        # 计算最终的损失，即所有样本损失的平均值
        loss = torch.mean(loss_per_sample)
        loss_record = {'Knowledge_Injection_Loss':loss.item()}

        return loss, loss_record
    
    """
    用传统的对比学习来注入知识（和get_loss2的作用其实是一样的，不一样的实现方式而已）
    """
    def get_loss2(self, ehr, et, PT1_score, PT2_score, mask):
        first_embeddings = et[:, 0, :].unsqueeze(1)
        et = et + torch.where(mask.unsqueeze(-1), first_embeddings, torch.zeros_like(et))
        
        similarity = F.cosine_similarity(ehr, et, dim=-1)
        similarity = similarity/self.ki_cl_tau
        similarity = F.softmax(similarity, dim=-1)
        
        positive_mask = mask.clone()
        positive_mask[:, 0] = 1  # 将每行的第一个元素设置为正例
        numerator = torch.sum(similarity * positive_mask, dim=1)
        
        loss = -torch.log(numerator)
        
        loss = torch.mean(loss)
        loss_record = {'Knowledge_Injection_Loss':loss.item()}

        return loss, loss_record
    
    """
    用层次对比学习来注入知识
    """
    def get_loss3(self, ehr, et, PT1_score, PT2_score, mask):
        first_embeddings = et[:, 0, :].unsqueeze(1)
        et = et + torch.where(mask.unsqueeze(-1), first_embeddings, torch.zeros_like(et))
        
        similarity = F.cosine_similarity(ehr, et, dim=-1)
        similarity = similarity/self.ki_cl_tau
        logits1 = F.softmax(similarity, dim=-1)
        
        # 正例的损失
        eps = 1e-8
        numerator = logits1[:, 0]
        loss1 = -torch.log(numerator+eps)
        loss1 = loss1.mean()
        
        # 弱正例损失,处理有弱正例的行
        logits2 = similarity.clone()[:, 1:]
        logits2 = F.softmax(logits2, dim=-1)
        weak_positive_mask = mask.clone()[:, 1:]
        
        valid_samples_mask = weak_positive_mask.sum(dim=-1) > 0
        logits2_valid = logits2[valid_samples_mask]
        weak_positive_mask_valid = weak_positive_mask[valid_samples_mask]
        
        numerator = torch.sum(logits2_valid * weak_positive_mask_valid, dim=-1)
        loss2 = -torch.log(numerator + eps)  # eps为小的正数，避免对0取对数
        loss2 = loss2.mean()  # 计算平均loss
        
        # 弱正例损失，处理没有弱正例的行
        logits3 = similarity.clone()[:, 1:]
        logits3 = torch.exp(logits3)
        weak_positive_mask = mask.clone()[:, 1:]
        
        invalid_samples_mask = weak_positive_mask.sum(dim=-1) < 1
        logits3_invalid = logits3[invalid_samples_mask]
        
        numerator = 1 / torch.sum(logits3_invalid, dim=-1)
        loss3 = -torch.log(numerator)
        loss3 = loss3.mean()
        
        # print(logits2_valid.shape, loss2)
        
        if weak_positive_mask.sum().item() < 1:
            loss = (0.5 * loss1)
        else:
            loss = (0.33 * loss1) + (0.33 * loss2)  + (0.33 * loss3)
        loss_record = {'Knowledge_Injection_Loss':loss.item()}
        
        return loss, loss_record
        
        
        
    def forward(self, ehr, et, PT1_score, PT2_score):
        mask = self.get_mask(ehr, et, PT1_score, PT2_score)
        loss, loss_record = self.get_loss3(ehr, et, PT1_score, PT2_score, mask)
        
        return loss, loss_record
        
        
class Sentence_score(nn.Module):
    def __init__(self, args):
        super(Sentence_score, self).__init__()      
        self.args = args
        self.target_dim = args.target_dim
        self.entity_dim = args.target_dim * args.entity_mul
        self.relation_dim = args.target_dim * args.relation_mul
        self.mul = 2
        
        self.encoder = nn.TransformerEncoderLayer(d_model=self.entity_dim, nhead=2, 
                                                  dim_feedforward=self.entity_dim*self.mul, dropout=0.1, activation='relu')
        self.MLP = nn.Sequential(
            nn.Linear(self.entity_dim, self.entity_dim*2),
            nn.LeakyReLU(),
            nn.Linear(self.entity_dim*2, 1),
        )
        self.CLS = nn.Parameter(torch.randn(self.entity_dim))
        
    def forward(self, eh, er, et):
        cls_expanded = self.CLS.unsqueeze(0).unsqueeze(0).expand_as(eh)
        et_reshaped = et.view(-1, 1, self.entity_dim)
        eh_expanded = eh.repeat(1, self.nneg + 1, 1).view(-1, 1, self.entity_dim)
        er_expanded = er.repeat(1, self.nneg + 1, 1).view(-1, 1, self.entity_dim)
        cls_expanded = cls_expanded.repeat(1, self.nneg + 1, 1).view(-1, 1, self.entity_dim)
        concatenated = torch.cat([cls_expanded, eh_expanded, er_expanded, et_reshaped], dim=1)  # [batch * (nneg+1), 4, dim]
        
        outputs = self.encoder(concatenated)
        semantic_outputs = outputs[:, 0, :]
        semantic_outputs = semantic_outputs.view(-1, self.nneg + 1, self.entity_dim)
        score = self.MLP(semantic_outputs)
        
        return score
        
        



class EntityBias(nn.Module):
    def __init__(self, args):
        super(EntityBias, self).__init__()
        self.args = args
        
        # 声明两个与 args.nentity 一样大的可学习的一维张量
        self.head_bias = nn.Parameter(torch.randn(args.nentity) * 0.01)
        self.tail_bias = nn.Parameter(torch.randn(args.nentity) * 0.01)

    def forward(self, data, stu_dis):    
        # 提取正样本和负样本
        positive_sample, negative_sample = data

        # 从正样本中获取 head 和 tail 的索引
        head_indices = positive_sample[:, 0]  # head 索引
        tail_indices = positive_sample[:, 2]  # tail 索引

        # 根据 head 和 tail 索引获取对应的 bias
        head_bias = self.head_bias[head_indices]  # [batch]
        tail_bias_pos = self.tail_bias[tail_indices]  # [batch]

        # 从负样本中获取 tail 索引并计算对应的 bias
        tail_indices_neg = negative_sample  # [batch, nneg]
        tail_bias_neg = self.tail_bias[tail_indices_neg]  # [batch, nneg]

        # 拼接正样本的 tail_bias 和负样本的 tail_bias
        tail_bias = torch.cat([tail_bias_pos.unsqueeze(1), tail_bias_neg], dim=1)  # [batch, nneg+1]

        # 将 head_bias 按行广播到 [batch, nneg+1]
        head_bias_broadcasted = head_bias.unsqueeze(1).expand(-1, stu_dis.size(1))  # [batch, nneg+1]

        # 与 stu_dis 相加
        stu_dis = head_bias_broadcasted + stu_dis
        stu_dis = tail_bias + stu_dis

        return stu_dis

        
        