#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import numpy as np
import torch
import time
import random
import os

from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size, args=None):
        self.args = args
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.count = self.count_frequency(triples)
        self.true_triples = self.get_true_head_and_tail(self.triples)
        self.data_type = torch.double if self.args.data_type=='double' else torch.float
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        # new_seed = int(time.time() * 1000) % 10000 
        # np.random.seed(new_seed) # 重置随机种子
        
        positive_sample = self.triples[idx]

        head, relation, tail = positive_sample
        
        # 出现次数少权重就大，出现次数多权重就少。
        subsampling_weight = self.count[(head, relation)]
        
        if relation >= self.nrelation/2: # 说明这个三元组是reverse之后的
            subsampling_weight += self.count[(tail, relation-int(self.nrelation/2))]
        else: # 说明这个三元组是reverse之前的
            subsampling_weight += self.count[(tail, relation+int(self.nrelation/2))]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        negative_sample_list = []
        negative_sample_size = 0
        
        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2)
            mask = np.in1d(
                negative_sample, 
                self.true_triples[(head, relation)], 
                # np.concatenate((self.true_triples[(head, relation)], self.m_neg[(head, relation)])),
                assume_unique=True, 
                invert=True #返回一个和negative_sampleU一样大的数组True表示negative_sample中不在self.true_head[(relation, tail)]中的元素
            )                         
            # 去除掉对应位置为False的样本
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        # 去掉多余的
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        np.random.shuffle(negative_sample)
        
        '''
        把head作为负样本手动的确定的加入进去
        '''
        if head not in self.true_triples[(head, relation)]:
            negative_sample[random.randint(0, self.negative_sample_size - 1)] = head
        
        negative_sample = torch.LongTensor(negative_sample)
        positive_sample = torch.LongTensor(positive_sample)
        # positive_sample返回的是正确的三元组，negative_sample返回的是替换的entity号，subsampling_weight返回的是对应正确三元组的权重
        return positive_sample, negative_sample, subsampling_weight
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        return positive_sample, negative_sample, subsample_weight
    
    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1
        return count
    
    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''
        true_triples = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_triples:
                true_triples[(head, relation)] = []
            true_triples[(head, relation)].append(tail)
        # 去重后返回
        for head, relation in true_triples:
            true_triples[(head, relation)] = np.array(list(set(true_triples[(head, relation)])))             

        return true_triples
    

'''
本类的主要功能是生成只包含负样本的训练数据集，用于构成双重负采样的训练数据
'''
class DoubleNegativeTrainDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size, args=None):
        self.args = args
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.count = self.count_frequency(triples)
        self.true_triples = self.get_true_head_and_tail(self.triples)
        self.data_type = torch.double if self.args.data_type=='double' else torch.float
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        # 给定的三元组 (head, relation, tail)
        positive_sample = self.triples[idx]
        head, relation, _ = positive_sample

        # 找到一个满足条件的 head1
        while True:
            head1 = np.random.randint(self.nentity)
            if (head1, relation) not in self.true_triples:
                break

        # 生成 self.negative_sample_size + 1 个不同的 tail
        tails = np.random.choice(self.nentity, self.negative_sample_size + 1, replace=False)

        # 将第一个 tail 作为 "positive_sample 的 tail" 返回
        positive_sample = torch.LongTensor([head1, relation, tails[0]])

        # 剩余的 tail 作为负样本
        negative_sample = torch.LongTensor(tails[1:])

        # subsampling_weight 始终为 1
        subsampling_weight = torch.tensor([1.0])

        # 返回新的 positive_sample、negative_sample 和 subsampling_weight
        return positive_sample, negative_sample, subsampling_weight
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        return positive_sample, negative_sample, subsample_weight
    
    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1
        return count
    
    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''
        true_triples = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_triples:
                true_triples[(head, relation)] = []
            true_triples[(head, relation)].append(tail)
        # 去重后返回
        for head, relation in true_triples:
            true_triples[(head, relation)] = np.array(list(set(true_triples[(head, relation)])))             

        return true_triples




class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]


        tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (-1, tail) for rand_tail in range(self.nentity)]
        # tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
        #            else (-1, rand_tail) for rand_tail in range(self.nentity)]
        tmp[tail] = (0, tail)
        # tmp[head] = (-1, tail)

            
        tmp = torch.LongTensor(tmp)            
        filter_bias = tmp[:, 0].float()
        
        negative_sample = tmp[:, 1]
        positive_sample = torch.LongTensor((head, relation, tail))

        return positive_sample, negative_sample, filter_bias
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        
        return positive_sample, negative_sample, filter_bias

"""
用来获取KGE分数最高的数据集
"""
class Fake_TestDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, args=None):
        self.args = args
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.count = self.count_frequency(triples)
        self.true_triples = self.get_true_head_and_tail(self.triples)
        self.data_type = torch.double if self.args.data_type=='double' else torch.float
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        # new_seed = int(time.time() * 1000) % 10000 
        # np.random.seed(new_seed) # 重置随机种子
        
        positive_sample = self.triples[idx]
        negative_sample = np.arange(self.nentity)
        subsampling_weight = torch.tensor(1.0)
        
        negative_sample = torch.LongTensor(negative_sample)
        positive_sample = torch.LongTensor(positive_sample)
        # positive_sample返回的是正确的三元组，negative_sample返回的是替换的entity号，subsampling_weight返回的是对应正确三元组的权重
        return positive_sample, negative_sample, subsampling_weight
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        return positive_sample, negative_sample, subsample_weight
    
    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1
        return count
    
    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''
        true_triples = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_triples:
                true_triples[(head, relation)] = []
            true_triples[(head, relation)].append(tail)
        # 去重后返回
        for head, relation in true_triples:
            true_triples[(head, relation)] = np.array(list(set(true_triples[(head, relation)])))             

        return true_triples
        


class BidirectionalOneShotIterator(object):
    def __init__(self, train_dataloader):
        self.train_dataloader = self.one_shot_iterator(train_dataloader)
        self.len = len(train_dataloader)
        
    def __next__(self):
        data = next(self.train_dataloader)
        
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data


"""
这个迭代器允许有两个风格不一致的训练数据集，并且交替的取出其中的数据
"""
class BidirectionalOneShotIterator2(object):
    def __init__(self, train_dataloader1, train_dataloader2):
        self.train_dataloader1 = self.one_shot_iterator(train_dataloader1)
        self.train_dataloader2 = self.one_shot_iterator(train_dataloader2)
        self.step = 0
        self.len = len(train_dataloader1) + len(train_dataloader2)
        
    def __next__(self):
        # 轮流替换头和尾
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.train_dataloader1)
        else:
            data = next(self.train_dataloader2)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data