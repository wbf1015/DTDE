import sys
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

original_directory = os.getcwd()
new_directory = original_directory + '/code/Decoder/Logits_Feature_Decoder/'
if new_directory not in sys.path:
    sys.path.append(new_directory)

from LFD_auxiliary_loss import *
from LFD_contrastive_learning import *
from LFD_fusion import *
from LFD_norm import *
from LFD_soft_loss import *
from LFD_hard_loss import *
from LFD_embedding_transform import *
from LFD_soft_loss_calculation import *
from LFD_similarity import *
from LFD_Encoder import *

sys.path.remove(new_directory)

class Decoder_2KGE(nn.Module):
    def __init__(self, args):
        super(Decoder_2KGE, self).__init__()
        self.args = args
        self.target_dim = args.target_dim
        self.entity_dim = args.target_dim * args.entity_mul
        self.relation_dim = args.target_dim * args.relation_mul

        # 硬损失函数
        self.hard_loss1 = SigmoidLoss(args=args)
        self.hard_loss2 = SigmoidLoss2(args=args)
        
        # 软损失函数
        self.soft_loss = Imitation_SingleTeacher(args=args, stu_score_preprocess='local_standardize', tea_score_preprocess='constant', distill_func='KL_divergency')
        # self.soft_loss2 = Imitation_SingleTeacher(args=args, stu_score_preprocess='constant', tea_score_preprocess='constant', distill_func='DistillKL_Logit_Standard')
        # self.soft_loss2 = Imitation_DualTeacherv2(args=args, stu_score_preprocess='local_standardize', tea_score_preprocess='constant', distill_func='KL_divergency', fusion_function='scores_fusionv1')
        self.soft_loss2 = Imitation_DualTeacherv3(args=args, stu_score_preprocess='local_standardize', tea_score_preprocess='constant', distill_func='KL_divergency', fusion_function='scores_fusionv7', fusion_scores_loss='SigmoidLoss')
        
        # Encoder组件
        # self.entitystage1Encoder = ConvE(args, input_dim=256, height_dim=16, width_dim=16, output_dim=64, output_channel=8)
        # self.relationstage1Encoder = ConvE(args, input_dim=256, height_dim=16, width_dim=16, output_dim=64, output_channel=8)
        # self.entitystage2Encoder = ConvE(args, input_dim=64, height_dim=8, width_dim=8, output_dim=32, output_channel=8)
        # self.relationstage2Encoder = ConvE(args, input_dim=64, height_dim=8, width_dim=8, output_dim=32, output_channel=8)
        
        self.entitystage1Encoder = Easy_MLP(args, input_dim=32, output_dim=64, layer_mul=2)
        self.relationstage1Encoder = Easy_MLP(args, input_dim=32, output_dim=64, layer_mul=2)
        self.entitystage2Encoder = Easy_MLP(args, input_dim=64, output_dim=128, layer_mul=2)
        self.relationstage2Encoder = Easy_MLP(args, input_dim=64, output_dim=128, layer_mul=2)
        
        # self.entitystage1Encoder = Transformer_Encoder(args, input_dim=512, output_dim=128, seq_len=4, n_head=4)
        # self.relationstage1Encoder = Transformer_Encoder(args, input_dim=256, output_dim=128, seq_len=4, n_head=4)
        # self.entitystage2Encoder = Transformer_Encoder(args, input_dim=128, output_dim=32, seq_len=4, n_head=1)
        # self.relationstage2Encoder = Transformer_Encoder(args, input_dim=128, output_dim=32, seq_len=4, n_head=1)
        
        # 辅助模块
        # self.Stage1EntitySemanticTransform = Easy_MLP(args, input_dim=32, output_dim=128, layer_mul=2)
        # self.Stage1RelationSemanticTransform = Easy_MLP(args, input_dim=32, output_dim=128, layer_mul=2)
        # self.Stage2EntitySemanticTransform = Easy_MLP(args, input_dim=128, output_dim=512, layer_mul=2)
        # self.Stage2RelationSemanticTransform = Easy_MLP(args, input_dim=128, output_dim=512, layer_mul=2)
        self.stage0weight_learner = weight_learnerv2(args, entity_dim=32, relation_dim=32)
        self.stage1weight_learner = weight_learnerv2(args, entity_dim=64, relation_dim=32)
        self.stage2weight_learner = weight_learnerv2(args, entity_dim=128, relation_dim=32)
        # self.align = MSEAlign(args, stu_dim0=32, stu_dim1=64, stu_dim2=128, tea_dim=512, semantic_dim=128, layermul=2, weight=1)
        # self.align = MSEAlignv2(args, stu_dim0=32, stu_dim1=64, stu_dim2=128, tea_dim=512, layermul=2, weight=1)
        self.align  = MSEAlignv4(args, stu_dim0=32, stu_dim1=64, stu_dim2=128, tea_dim1=512, tea_dim2=512, hidden_dim=128, layermul=2, weight=1)
        self.cl_module = ContrastiveLossv3(args, stu_dim0=32, stu_dim1=64, stu_dim2=128, tea_dim1=512, tea_dim2=512, semantic_dim=128, layermul=1)
        self.Knowledge_Injection = Knowledge_Injection(args)
        
        # 一些必要的组件
        self.cal_sim = cal_similarity(args=self.args, temperature=self.args.temprature)
        self.stage0Combine_hr = Combine_hr(entity_dim=self.entity_dim, relation_dim=self.relation_dim, hidden_dim=self.target_dim, layer_mul=2)
        self.stage0tail_transform = BN(input_dim=self.target_dim)
        self.stage1Combine_hr = Combine_hr(entity_dim=64, relation_dim=64, hidden_dim=64, layer_mul=2)
        self.stage1tail_transform = BN(input_dim=64)
        self.stage2Combine_hr = Combine_hr(entity_dim=128, relation_dim=128, hidden_dim=128, layer_mul=2)
        self.stage2tail_transform = BN(input_dim=128)
    
    def get_student_score(self, ehr, et):        
        stu_score = self.cal_sim.SCCF_similarity3(ehr, et)
        stu_score[:, 0] -= self.args.pos_gamma  # 第一列减去pos_gamma
        stu_score[:, 1:] -= self.args.neg_gamma  # 其余列减去neg_gamma
        
        return stu_score
    
    def basic_loss(self, ehr, et, PT1_score, PT2_score, weight_mask, data=None, Teacher_embeddings=None, subsampling_weight=None, prefix=''):
        stu_score= self.get_student_score(ehr, et)
        
        hard_loss, loss_record = self.hard_loss1(stu_score, subsampling_weight)
        hard_loss2, loss_record2 = self.hard_loss2(stu_score, subsampling_weight)
        
        soft_loss1, soft_loss_record1 = self.soft_loss(stu_score, PT1_score, prefix='RotatE')
        soft_loss2, soft_loss_record2 = self.soft_loss(stu_score, PT2_score, prefix='LorentzKG')
        
        loss = weight_mask[0] * hard_loss
        loss += weight_mask[1] * hard_loss2
        loss += weight_mask[2] * self.args.kdloss_weight * soft_loss1 
        loss += weight_mask[3] * self.args.kdloss_weight * soft_loss2
        
        loss_record.update(loss_record)
        loss_record.update(loss_record2)
        loss_record.update(soft_loss_record1)
        loss_record.update(soft_loss_record2)
        loss_record.update({'LOSS':loss.item()})
        
        loss_record = {prefix + key: value for key, value in loss_record.items()}
        return loss, loss_record
    
    def loss_ensemble(self, ehr, et, PT1_score, PT2_score, weight_mask, data=None, Teacher_embeddings=None, subsampling_weight=None, prefix=''):
        stu_score= self.get_student_score(ehr, et)
        
        if prefix == 'Stage0_':
            weight = self.stage0weight_learner(ehr, et, data)
        if prefix == 'Stage1_':
            weight = self.stage1weight_learner(ehr, et, data)
        if prefix == 'Stage2_':
            weight = self.stage2weight_learner(ehr, et, data)
        
        hard_loss, loss_record = self.hard_loss1(stu_score, subsampling_weight)
        hard_loss2, loss_record2 = self.hard_loss2(stu_score, subsampling_weight)
        
        soft_loss1, soft_loss_record1 = self.soft_loss(stu_score, PT1_score, prefix='RotatE')
        soft_loss2, soft_loss_record2 = self.soft_loss(stu_score, PT2_score, prefix='LorentzKG')
        soft_lossfusion, soft_loss_recordfusion = self.soft_loss2(stu_score, PT1_score, PT2_score, prefix='Fusion', weight=weight)
        
        # kiloss, kiloss_record = self.Knowledge_Injection(ehr, et, PT1_score, PT2_score)
        
        loss = weight_mask[0] * hard_loss
        loss += weight_mask[1] * hard_loss2
        loss += weight_mask[2] * self.args.kdloss_weight * soft_loss1 
        loss += weight_mask[3] * self.args.kdloss_weight * soft_loss2
        loss += weight_mask[4] * self.args.kdloss_weight * soft_lossfusion
        # loss += 0.1 * kiloss

        loss_record.update(loss_record)
        loss_record.update(loss_record2)
        loss_record.update(soft_loss_record1)
        loss_record.update(soft_loss_record2)
        loss_record.update(soft_loss_recordfusion)
        # loss_record.update(kiloss_record)
        loss_record.update({'Loss':loss.item()})
        
        loss_record = {prefix + key: value for key, value in loss_record.items()}
        return loss, loss_record
    
    def stage0_Encoder(self, eh, er, et, PT1_score, PT2_score, data=None, Teacher_embeddings=None, subsampling_weight=None):
        ehr = self.stage0Combine_hr(eh, er)
        et = self.stage0tail_transform(et)
        
        loss, loss_record = self.loss_ensemble(ehr, et, PT1_score, PT2_score, [0, 1, 0, 1, 0.5], data=data, Teacher_embeddings=Teacher_embeddings, subsampling_weight=subsampling_weight, prefix='Stage0_')
        
        return loss, loss_record
    
    def stage1_Encoder(self, eh, er, et, PT1_score, PT2_score, data=None, Teacher_embeddings=None, subsampling_weight=None):
        # eh, er, et = self.Stage1EntitySemanticTransform(eh), self.Stage1RelationSemanticTransform(er), self.Stage1EntitySemanticTransform(et)
        eh, er, et = self.entitystage1Encoder(eh), self.relationstage1Encoder(er), self.entitystage1Encoder(et)
        # eh, er, et = eh + eh_, er + er_, et + et_
        
        ehr = self.stage1Combine_hr(eh, er)
        et = self.stage1tail_transform(et)
        
        loss, loss_record = self.loss_ensemble(ehr, et, PT1_score, PT2_score, [0, 1, 0, 1, 0.5], data=data, Teacher_embeddings=Teacher_embeddings, subsampling_weight=subsampling_weight, prefix='Stage1_')
        
        return loss, loss_record, eh, er, et
    
    
    def stage2_Encoder(self, eh, er, et, PT1_score, PT2_score, data=None, Teacher_embeddings=None, subsampling_weight=None):
        # eh, er, et = self.Stage2EntitySemanticTransform(eh), self.Stage2RelationSemanticTransform(er), self.Stage2EntitySemanticTransform(et)
        eh, er, et = self.entitystage2Encoder(eh), self.relationstage2Encoder(er), self.entitystage2Encoder(et)
        # eh, er, et = eh + eh_, er + er_, et + et_
        
        ehr = self.stage2Combine_hr(eh, er)
        et = self.stage2tail_transform(et)
        
        loss, loss_record = self.loss_ensemble(ehr, et, PT1_score, PT2_score, [0, 1, 0, 1, 0.5], data=data, Teacher_embeddings=Teacher_embeddings, subsampling_weight=subsampling_weight, prefix='Stage2_')
        
        return loss, loss_record, eh, er, et
    
    
    def query_sample_batch(self, eh, er, et, PT1_score, PT2_score, data=None, Teacher_embeddings=None, subsampling_weight=None):
        PT_head1, PT_relation1, PT_tail1, PT_head2, PT_relation2, PT_tail2 = Teacher_embeddings
        loss0, loss_record0 = self.stage0_Encoder(eh, er, et, PT1_score, PT2_score, data=data, Teacher_embeddings=Teacher_embeddings, subsampling_weight=subsampling_weight)
        loss1, loss_record1, eh1, er1, et1 = self.stage1_Encoder(eh, er, et, PT1_score, PT2_score, data=data, Teacher_embeddings=Teacher_embeddings, subsampling_weight=subsampling_weight)
        loss2, loss_record2, eh2, er2, et2 = self.stage2_Encoder(eh1, er1, et1, PT1_score, PT2_score, data=data, Teacher_embeddings=Teacher_embeddings, subsampling_weight=subsampling_weight)
        
        # 辅助损失
        # alignloss, alignlossrecord = self.align(et, et1, et2, PT_tail2)
        
        # alignloss, alignlossrecord = self.align(et, et1, et2, PT_tail1, PT_tail2)
        alignloss, alignlossrecord = self.align(eh, eh1, eh2, PT_head1, PT_head2)
        # contrastiveloss, contrastivelossrecord = self.cl_module(et, et1, et2, PT_tail1, PT_tail2)
        contrastiveloss, contrastivelossrecord = self.cl_module(eh, eh1, eh2, PT_head1, PT_head2)
        
        loss = 0
        loss += 1 * loss0
        loss += 0.5 * loss1
        loss += 0.1 * loss2
        loss += alignloss
        loss += self.args.contrastive_weight * contrastiveloss
        
        loss_record = {}
        loss_record.update(loss_record0)
        loss_record.update(loss_record1)
        loss_record.update(loss_record2)
        loss_record.update(alignlossrecord)
        loss_record.update(contrastivelossrecord)
        loss_record.update({'loss':loss.item()})
        
        return loss, loss_record

    
    def random_sample_batch(self, eh, er, et, PT1_score, PT2_score, data=None, Teacher_embeddings=None, subsampling_weight=None):
        PT_head1, PT_relation1, PT_tail1, PT_head2, PT_relation2, PT_tail2 = Teacher_embeddings
        loss0, loss_record0 = self.stage0_Encoder(eh, er, et, PT1_score, PT2_score, data=data, Teacher_embeddings=Teacher_embeddings, subsampling_weight=subsampling_weight)
        loss1, loss_record1, eh1, er1, et1 = self.stage1_Encoder(eh, er, et, PT1_score, PT2_score, data=data, Teacher_embeddings=Teacher_embeddings, subsampling_weight=subsampling_weight)
        loss2, loss_record2, eh2, er2, et2 = self.stage2_Encoder(eh1, er1, et1, PT1_score, PT2_score, data=data, Teacher_embeddings=Teacher_embeddings, subsampling_weight=subsampling_weight)
        
        contrastiveloss, contrastivelossrecord = self.cl_module(et, et1, et2, PT_tail1, PT_tail2)
        
        loss = 0
        loss += 1 * loss0
        loss += 0.5 * loss1
        loss += 0.1 * loss2
        loss += self.args.contrastive_weight * contrastiveloss
        
        loss_record = {}
        loss_record.update(loss_record0)
        loss_record.update(loss_record1)
        loss_record.update(loss_record2)
        loss_record.update(contrastivelossrecord)
        loss_record.update({'loss':loss.item()})
        
        return loss, loss_record
        
    
    
    def forward(self, eh, er, et, PT1_score, PT2_score, data=None, Teacher_embeddings=None, subsampling_weight=None, mode='1posNneg'):
        if mode=='RandomSample':
            loss, loss_record = self.random_sample_batch(eh, er, et, PT1_score, PT2_score, data=data, Teacher_embeddings=Teacher_embeddings, subsampling_weight=subsampling_weight)
        
        if mode=='QueryAwareSample':
            loss, loss_record = self.query_sample_batch(eh, er, et, PT1_score, PT2_score, data=data, Teacher_embeddings=Teacher_embeddings, subsampling_weight=subsampling_weight)
    
        return loss, loss_record
    
    def predict(self, eh, er, et, PT1_score, PT2_score):
        score = self.soft_loss2.fusion_score(PT1_score, PT2_score)
        return score
    
    def predict2(self, eh, er, et):
        ehr = self.stage0Combine_hr(eh, er)
        et = self.stage0tail_transform(et)
        stu_score = self.cal_sim.SCCF_similarity3(ehr, et)
        return stu_score

    def predict3(self, eh, er, et):
        eh, er, et = self.entitystage1Encoder(eh), self.relationstage1Encoder(er), self.entitystage1Encoder(et)
        
        ehr = self.stage1Combine_hr(eh, er)
        et = self.stage1tail_transform(et)
        stu_score = self.cal_sim.SCCF_similarity3(ehr, et)
        return stu_score
    
    def predict4(self, eh, er, et):
        eh, er, et = self.entitystage1Encoder(eh), self.relationstage1Encoder(er), self.entitystage1Encoder(et)
        eh, er, et = self.entitystage2Encoder(eh), self.relationstage2Encoder(er), self.entitystage2Encoder(et)
        
        ehr = self.stage2Combine_hr(eh, er)
        et = self.stage2tail_transform(et)
        stu_score = self.cal_sim.SCCF_similarity3(ehr, et)
        return stu_score