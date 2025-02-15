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

from SD_auxiliary_loss import *
from SD_fusion import *
from SD_norm import *
from SD_soft_loss import *
from SD_similarity import *

class DistanceImitation_SingleTeacher(nn.Module):
    def __init__(self, args, stu_score_preprocess='constant', tea_score_preprocess='constant', distill_func='KL_divergency'):
        super(DistanceImitation_SingleTeacher, self).__init__()
        self.args = args
        self.target_dim = args.target_dim
        self.entity_dim = args.target_dim * args.entity_mul
        self.relation_dim = args.target_dim * args.relation_mul
        self.layer_mul = 2
        
        self.cal_sim = cal_similarity(args=self.args, temperature=self.args.temprature)
        
        self.stu_score_preprocess = globals()[stu_score_preprocess]
        self.tea_score_preprocess = globals()[tea_score_preprocess]
        
        if distill_func == 'HuberLoss':
            self.distill_func = globals()[distill_func](self.args)
        if distill_func == 'KL_divergency':
            self.distill_func = globals()[distill_func](self.args)
        if distill_func == 'KL_divergency2':
            self.distill_func = globals()[distill_func](self.args)
            
        self.query_scale = nn.Sequential(
            nn.Linear((2*self.entity_dim + self.relation_dim), (2*self.entity_dim + self.relation_dim) * self.layer_mul),
            nn.LeakyReLU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear((2*self.entity_dim + self.relation_dim) * self.layer_mul, 1)
        )
        
        self.query_scale2 = nn.Sequential(
            nn.Linear((self.entity_dim), (self.entity_dim) * self.layer_mul),
            nn.LeakyReLU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear((self.entity_dim) * self.layer_mul, 1)
        )
        
        self.tail_scale = nn.Sequential(
            nn.Linear((self.entity_dim), (self.entity_dim) * self.layer_mul),
            nn.LeakyReLU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear((self.entity_dim) * self.layer_mul) * self.layer_mul, 1
        )

    
    def scale_transe(self, eh, er, et, ehr_, et_, data):
        eh, er, et, ehr_, et_ = eh.detach(), er.detach(), et.detach(), ehr_.detach(), et_.detach()
        ehr_scale = self.query_scale(torch.cat((eh,er,ehr_),dim=-1))
        ehr_scale = self.query_scale2(ehr_)
        et_scale = self.tail_scale(et_)
        ehr_ = ehr_ * ehr_scale
        et_ = et_ * et_scale
        transe = et_-ehr_
        stu_dis = torch.norm(transe, p=2, dim=-1)
        stu_dis = -1*stu_dis
        return stu_dis
    
    def get_stu_dis_similar_triangles(self, eh, er, et, ehr_, et_, data):
        ehr_norm = torch.norm(ehr_, p=2, dim=-1, keepdim=True)
        et_norm = torch.norm(et_, p=2, dim=-1, keepdim=True)
        ehr_normalize = ehr_/ehr_norm
        et_normalize = et_/et_norm
        chord = (et_normalize-ehr_normalize)*ehr_norm
        transe = et_-ehr_
        distance = chord-transe
        stu_dis = torch.norm(distance, p=2, dim=-1)
        stu_dis = -1*stu_dis
        return stu_dis
    
    def get_stu_dis_transe(self, eh, er, et, ehr_, et_, data):
        transe = ehr_-et_
        stu_dis = torch.norm(transe, p=2, dim=-1)
        stu_dis = 9.0-stu_dis
        
        return stu_dis
        
    def forward(self, eh, er, et, ehr_, et_, data, tea_dis, prefix=''):
        
        stu_dis = self.get_stu_dis(eh, er, et, ehr_, et_, data)
    
        stu_dis, _, _ = self.stu_score_preprocess(stu_dis)
        tea_dis, _, _ = self.tea_score_preprocess(tea_dis)        
        
        if type(self.distill_func).__name__ in ['HuberLoss', 'KL_divergency']:
            loss = self.distill_func(stu_dis, tea_dis)
        if type(self.distill_func).__name__ in ['KL_divergency2']:
            mask = only_learn_right_mask(stu_dis, tea_dis)
            loss = self.distill_func(stu_dis, tea_dis, mask)
        
        loss_record = {
            'soft_loss'+prefix: loss.item(),
        }
        
        return loss, loss_record




def only_learn_right(student_dis, teacher_dis):
    reference_scores = teacher_dis[:, 0].unsqueeze(1)
    
    # Mask for values greater than the reference score
    mask = teacher_dis > reference_scores
    
    # Set values greater than the reference to a large negative value (-1e9)
    modified_teacher_dis = torch.where(mask, torch.tensor(-1e3, device=teacher_dis.device), teacher_dis)
    modified_student_dis = torch.where(mask, torch.tensor(-1e3, device=student_dis.device), student_dis)
    
    return modified_student_dis, modified_teacher_dis

def only_learn_right_mask(student_dis, teacher_dis):
    reference_scores = teacher_dis[:, 0].unsqueeze(1)
    mask = teacher_dis > reference_scores
    
    return mask

def record_rank_4_case_study(PT1_dis, PT2_dis, PT_dis, record_path):
    with open(record_path, "w") as f:
        # Iterate through each batch
        for batch_idx in range(PT1_dis.size(0)):
            # Get the current batch scores
            PT1_scores = PT1_dis[batch_idx]
            PT2_scores = PT2_dis[batch_idx]
            PT_scores = PT_dis[batch_idx]

            # Get the indices of the sorted scores (from high to low)
            PT1_sorted_indices = torch.argsort(PT1_scores, descending=True)
            PT2_sorted_indices = torch.argsort(PT2_scores, descending=True)
            PT_sorted_indices = torch.argsort(PT_scores, descending=True)

            # Sort the scores based on the indices
            PT1_sorted_scores = PT1_scores[PT1_sorted_indices]
            PT2_sorted_scores = PT2_scores[PT2_sorted_indices]
            PT_sorted_scores = PT_scores[PT_sorted_indices]

            # Apply softmax to the sorted scores
            PT1_softmax = F.softmax(PT1_sorted_scores, dim=0)
            PT2_softmax = F.softmax(PT2_sorted_scores, dim=0)
            PT_softmax = F.softmax(PT_sorted_scores, dim=0)

            # Write the sorted indices, original scores, and softmax values to the file
            f.write("PT1 sorted indices: " + " ".join(map(str, PT1_sorted_indices.tolist())) + "\n")
            f.write("PT1 original scores: " + " ".join(map(lambda x: f"{x:.4f}", PT1_sorted_scores.tolist())) + "\n")
            f.write("PT1 softmax values: " + " ".join(map(lambda x: f"{x:.4f}", PT1_softmax.tolist())) + "\n")
            
            f.write("PT2 sorted indices: " + " ".join(map(str, PT2_sorted_indices.tolist())) + "\n")
            f.write("PT2 original scores: " + " ".join(map(lambda x: f"{x:.4f}", PT2_sorted_scores.tolist())) + "\n")
            f.write("PT2 softmax values: " + " ".join(map(lambda x: f"{x:.4f}", PT2_softmax.tolist())) + "\n")
            
            f.write("PT sorted indices: " + " ".join(map(str, PT_sorted_indices.tolist())) + "\n")
            f.write("PT original scores: " + " ".join(map(lambda x: f"{x:.4f}", PT_sorted_scores.tolist())) + "\n")
            f.write("PT softmax values: " + " ".join(map(lambda x: f"{x:.4f}", PT_softmax.tolist())) + "\n")

            # Add two blank lines after each batch
            f.write("\n\n")

def record_rank_4_case_study2(PT1_dis, PT2_dis, PT_dis, record_path):
    with open(record_path, "w") as f:
        # Iterate through each batch
        for batch_idx in range(PT1_dis.size(0)):
            # Get the current batch scores
            PT1_scores = PT1_dis[batch_idx]
            PT2_scores = PT2_dis[batch_idx]
            PT_scores = PT_dis[batch_idx]

            # Get the indices of the sorted scores (from high to low)
            PT1_sorted_indices = torch.argsort(PT1_scores, descending=True)
            PT2_sorted_indices = torch.argsort(PT2_scores, descending=True)
            PT_sorted_indices = torch.argsort(PT_scores, descending=True)

            # Sort the scores based on the indices
            PT1_sorted_scores = PT1_scores[PT1_sorted_indices]
            PT2_sorted_scores = PT2_scores[PT2_sorted_indices]
            PT_sorted_scores = PT_scores[PT_sorted_indices]

            PT1_sorted_scores_relative = -1 * ((PT1_sorted_scores[0] - PT1_sorted_scores) / torch.abs(PT1_sorted_scores[0] + 1e-8))
            PT2_sorted_scores_relative = -1 * ((PT2_sorted_scores[0] - PT2_sorted_scores) / torch.abs(PT2_sorted_scores[0] + 1e-8))
            PT_sorted_scores_relative = -1 * ((PT_sorted_scores[0] - PT_sorted_scores) / torch.abs(PT_sorted_scores[0] + 1e-8))

            PT1_softmax = F.softmax(PT1_sorted_scores, dim=0)
            PT2_softmax = F.softmax(PT2_sorted_scores, dim=0)
            PT_softmax = F.softmax(PT_sorted_scores, dim=0)

            PT1_softmax_relative = F.softmax(PT1_sorted_scores_relative, dim=0)
            PT2_softmax_relative = F.softmax(PT2_sorted_scores_relative, dim=0)
            PT_softmax_relative = F.softmax(PT_sorted_scores_relative, dim=0)

            # Write the sorted indices, original scores, relative scores, and softmax values to the file
            f.write("PT1 sorted indices: " + " ".join(map(str, PT1_sorted_indices.tolist())) + "\n")
            f.write("PT1 original scores: " + " ".join(map(lambda x: f"{x:.4f}", PT1_sorted_scores.tolist())) + "\n")
            f.write("PT1 relative scores: " + " ".join(map(lambda x: f"{x:.4f}", PT1_sorted_scores_relative.squeeze().tolist())) + "\n")
            f.write("PT1 softmax values: " + " ".join(map(lambda x: f"{x:.4f}", PT1_softmax.tolist())) + "\n")
            f.write("PT1 softmax relative values: " + " ".join(map(lambda x: f"{x:.4f}", PT1_softmax_relative.tolist())) + "\n")
            f.write("\n")
            
            f.write("PT2 sorted indices: " + " ".join(map(str, PT2_sorted_indices.tolist())) + "\n")
            f.write("PT2 original scores: " + " ".join(map(lambda x: f"{x:.4f}", PT2_sorted_scores.tolist())) + "\n")
            f.write("PT2 relative scores: " + " ".join(map(lambda x: f"{x:.4f}", PT2_sorted_scores_relative.squeeze().tolist())) + "\n")
            f.write("PT2 softmax values: " + " ".join(map(lambda x: f"{x:.4f}", PT2_softmax.tolist())) + "\n")
            f.write("PT2 softmax relative values: " + " ".join(map(lambda x: f"{x:.4f}", PT2_softmax_relative.tolist())) + "\n")
            f.write("\n")
            
            f.write("PT sorted indices: " + " ".join(map(str, PT_sorted_indices.tolist())) + "\n")
            f.write("PT original scores: " + " ".join(map(lambda x: f"{x:.4f}", PT_sorted_scores.tolist())) + "\n")
            f.write("PT relative scores: " + " ".join(map(lambda x: f"{x:.4f}", PT_sorted_scores_relative.squeeze().tolist())) + "\n")
            f.write("PT softmax values: " + " ".join(map(lambda x: f"{x:.4f}", PT_softmax.tolist())) + "\n")
            f.write("PT softmax relative values: " + " ".join(map(lambda x: f"{x:.4f}", PT_softmax_relative.tolist())) + "\n")

            # Add two blank lines after each batch
            f.write("\n\n\n")