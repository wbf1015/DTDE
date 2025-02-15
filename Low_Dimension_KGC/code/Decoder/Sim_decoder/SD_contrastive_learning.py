import sys
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class ContrastiveLoss(nn.Module):
    def __init__(self, args):
        """
        Contrastive Loss implementation.

        Args:
            tau (float): Temperature parameter for scaling.
        """
        super(ContrastiveLoss, self).__init__()
        self.args = args
        self.tau  = self.args.contrastive_tau

    def forward(self, output, target):
        """
        Args:
            zi (torch.Tensor): Positive example tensor with shape [batch, 1, dim].
            zj (torch.Tensor): Negative example tensor with shape [batch, nneg, dim].
        
        Returns:
            torch.Tensor: The computed loss value.
        """
        target = torch.zeros(output.shape[0]).long().to(output.device)
        output = output/self.tau
        contrastive_loss = F.cross_entropy(output, target, reduction="mean")
        loss = contrastive_loss
        loss_record = {
            'contrastive_loss':loss
        }

        return loss, loss_record
    
    def forward2(self, ehr, et):
        if ehr.shape[1] < et.shape[1]: 
            ehr = ehr.expand(-1, et.shape[1], -1)
        else:
            et = et.expand(-1, ehr.shape[1], -1)
        similarity = F.cosine_similarity(ehr, et, dim=-1)
        similarity = similarity/self.tau
        similarity = torch.exp(similarity)
        
        target = torch.zeros(similarity.shape[0]).long().to(similarity.device)
        contrastive_loss = F.cross_entropy(similarity, target, reduction="mean")
        loss = contrastive_loss
        loss_record = {
            'contrastive_loss':loss
        }
        return loss, loss_record



class ContrastiveLossv2(nn.Module):
    def __init__(self, args):
        """
        Contrastive Loss implementation.

        Args:
            tau (float): Temperature parameter for scaling.
        """
        super(ContrastiveLossv2, self).__init__()
        self.args = args
        self.tau  = self.args.contrastive_tau
        self.teacher_embedding = self.args.input_dim
        self.hidden_dim = 128
        self.layermul = 2
        self.entity_dim = args.target_dim * args.entity_mul
        self.relation_dim = args.target_dim * args.relation_mul
        
        self.Teacher1MLP = nn.Sequential(
            nn.Linear(self.teacher_embedding, self.hidden_dim * self.layermul),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim * self.layermul, self.hidden_dim),
        )
        
        self.Teacher2MLP = nn.Sequential(
            nn.Linear(self.teacher_embedding, self.hidden_dim * self.layermul),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim * self.layermul, self.hidden_dim),
        )
        
        self.StudentMLP = nn.Sequential(
            nn.Linear(self.entity_dim, self.hidden_dim * self.layermul),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim * self.layermul, self.hidden_dim),
        )
    
    """
    Input Shape = [batch, 1, embedding_dim]
    """
    def contrastive_similarity(self, stu_embedding, tea_embedding):
        stu_embedding = stu_embedding.squeeze(1)  # shape: [batch, embedding_dim]
        tea_embedding = tea_embedding.squeeze(1)  # shape: [batch, embedding_dim]

        stu_embedding = torch.nn.functional.normalize(stu_embedding, p=2, dim=1)
        tea_embedding = torch.nn.functional.normalize(tea_embedding, p=2, dim=1)

        cosine_similarity_matrix = torch.matmul(stu_embedding, tea_embedding.T)
        cosine_similarity_matrix = cosine_similarity_matrix / self.tau
        
        softmax_score = F.log_softmax(cosine_similarity_matrix, dim=1)  # shape: [batch, batch]
        labels = torch.arange(cosine_similarity_matrix.size(0)).to(stu_embedding.device)  # shape: [batch]
        loss = F.nll_loss(softmax_score, labels)  # 使用负对数似然损失 (相当于交叉熵)

        return loss
    
    """
    Input Shape = [batch, nneg+1, embedding_dim]
    """
    def contrastive_similarityv2(self, stu_embedding, tea_embedding):
        # 归一化嵌入向量
        stu_embedding = F.normalize(stu_embedding, p=2, dim=2)  # shape: [batch, nneg+1, embedding_dim]
        tea_embedding = F.normalize(tea_embedding, p=2, dim=2)  # shape: [batch, nneg+1, embedding_dim]

        # 计算余弦相似度矩阵
        cosine_similarity_matrix = torch.bmm(stu_embedding, tea_embedding.transpose(1, 2))  # shape: [batch, nneg+1, nneg+1]
        cosine_similarity_matrix = cosine_similarity_matrix / self.tau

        # 应用 softmax 得到每个 batch 中的概率分布
        softmax_score = F.log_softmax(cosine_similarity_matrix, dim=2)  # shape: [batch, nneg+1, nneg+1]

        # 创建标签，每个 batch 的第 i 个元素的标签是 i
        labels = torch.arange(stu_embedding.size(1)).to(stu_embedding.device)  # shape: [nneg+1]
        labels = labels.unsqueeze(0).repeat(stu_embedding.size(0), 1)  # shape: [batch, nneg+1]

        
        # 计算损失，由于标签和输出都是 batch-wise，需要在对应的维度上应用 nll_loss
        # 将 softmax_score 和 labels reshape 为适合 nll_loss 的形状
        loss = F.nll_loss(softmax_score.view(-1, softmax_score.size(-1)), labels.view(-1), reduction='mean')
        return loss
    
    def forward(self, eh, er, et, Teacher_embeddings):
        PT_head1, PT_relation1, PT_tail1, PT_head2, PT_relation2, PT_tail2 = Teacher_embeddings
    
        stu_head, tea_head1, tea_head2 = self.StudentMLP(eh), self.Teacher1MLP(PT_head1), self.Teacher2MLP(PT_head2)
        stu_tail, tea_tail1, tea_tail2 = self.StudentMLP(et), self.Teacher1MLP(PT_tail1), self.Teacher2MLP(PT_tail2)
        
        # loss1 = self.contrastive_similarity(stu_head, tea_head1)
        loss2 = self.contrastive_similarity(stu_head, tea_head2)
        # loss3 = self.contrastive_similarityv2(stu_tail, tea_tail1)
        loss4 = self.contrastive_similarityv2(stu_tail, tea_tail2)
        
        loss = 0
        # loss += 0 * loss1
        loss += 1 * loss2
        # loss += 0 * loss3
        loss += 1 * loss4
        
        loss_record = {}
        # loss_record.update({'head_teacher1_contrastiveLoss' : loss1.item()})
        loss_record.update({'head_teacher2_contrastiveLoss' : loss2.item()})
        # loss_record.update({'tail_teacher1_contrastiveLoss' : loss3.item()})
        loss_record.update({'tail_teacher2_contrastiveLoss' : loss4.item()})


        return loss, loss_record        