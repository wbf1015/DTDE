import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import  Parameter

class ConvE(nn.Module):
    def __init__(self, args, input_dim=256, height_dim=16, width_dim=16, output_dim=64, output_channel=32):
        super(ConvE, self).__init__()
        self.args = args
        
        self.input_dropout = 0.1
        self.hide_dropout = 0.1
        self.feature_dropout = 0.1
        
        self.input_channel = 1
        self.output_channel = output_channel
        self.kernel_size = 3
        self.stride = 1
        self.padding = 0
        
        self.input_dim = input_dim
        self.height_dim = height_dim
        self.width_dim = width_dim
        self.output_dim = output_dim
        assert self.input_dim == (self.height_dim * self.width_dim)
        
        self.input_drop = nn.Dropout(self.input_dropout)
        self.hide_drop = nn.Dropout(self.hide_dropout)
        self.feature_drop = nn.Dropout2d(self.feature_dropout)
        
        self.conv = nn.Conv2d(self.input_channel, self.output_channel, (self.kernel_size, self.kernel_size), stride=self.stride, padding=self.padding, bias=True)
        self.bn0 = nn.BatchNorm2d(self.input_channel)
        self.bn1 = nn.BatchNorm2d(self.output_channel)
        self.bn2 = nn.BatchNorm1d(self.output_dim)
        
        self.fc = nn.Linear(self.conv_output_dim(self.stride, self.padding), self.output_dim)

    def conv_output_shape(self, stride=1, padding=0):
        output_height = (self.height_dim - self.kernel_size + 2 * padding) // stride + 1
        output_width = (self.width_dim - self.kernel_size + 2 * padding) // stride + 1
        return self.output_channel, output_height, output_width

    def conv_output_dim(self, stride=1, padding=0):
        output_height = (self.height_dim - self.kernel_size + 2 * padding) // stride + 1
        output_width = (self.width_dim - self.kernel_size + 2 * padding) // stride + 1
        return self.output_channel * output_height * output_width

    def forward(self, obj_embedding):
        batch, neg_sample = obj_embedding.shape[0], obj_embedding.shape[1]
        conv_input = obj_embedding.view(batch, neg_sample, self.height_dim, self.width_dim)
        conv_input = obj_embedding.view(batch*neg_sample, 1, self.height_dim, self.width_dim)
        conv_input = self.bn0(conv_input)
        x = self.input_drop(conv_input)
        x = self.conv(conv_input)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(x.shape[0], -1)#bacth*hide_size(38*8*32 = 9728)
        x = self.fc(x)
        x = self.hide_drop(x)
        x = self.bn2(x)
        # x = F.relu(x)#batch*dim          ent_ems.weight   dim*ent_num
        output = x.view(batch, neg_sample, self.output_dim)
        
        return output
        


class Easy_MLP(nn.Module):
    def __init__(self, args, input_dim=256, output_dim=64, layer_mul=2):
        super(Easy_MLP, self).__init__()
        self.args = args
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_mul = layer_mul
        
        self.MLP = nn.Sequential(
            nn.Linear((input_dim), int((input_dim) * self.layer_mul)),
            nn.LeakyReLU(),  # Uncomment this line if ReLU is needed after the first layer
            nn.Linear(int((input_dim) * self.layer_mul), output_dim),
            nn.BatchNorm1d(output_dim)  # Batch normalization on hidden_dim dimension
        )

    def forward(self, obj_embedding):
        batch, neg_sample, _ = obj_embedding.shape
        obj_embedding = obj_embedding.view(batch*neg_sample, -1)
        output = self.MLP(obj_embedding)
        output = output.view(batch, neg_sample, -1)
        return output



class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_diff, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_diff)
        self.fc2 = nn.Linear(d_diff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class Transformer_Encoder(nn.Module):
    def __init__(self, args, input_dim, output_dim, seq_len, n_head):
        super(Transformer_Encoder, self).__init__()
        self.args = args
        self.batch = None
        self.neg_sampling = None
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layermul = 2
        self.reduce_rate = self.input_dim//self.output_dim
        
        self.seq_len = seq_len
        self.n_head = n_head
        
        self.w_q = nn.Linear(self.input_dim//self.seq_len, (self.input_dim//self.seq_len)//self.reduce_rate)
        self.w_k = nn.Linear(self.input_dim//self.seq_len, (self.input_dim//self.seq_len)//self.reduce_rate)
        self.w_v = nn.Linear(self.input_dim//self.seq_len, (self.input_dim//self.seq_len)//self.reduce_rate)
        self.fc1 = nn.Linear(self.output_dim, self.output_dim)
        
        self.softmax = nn.Softmax(dim = -1)
        self.LayerNorm1 = nn.LayerNorm(self.output_dim).cuda()
        self.LayerNorm2 = nn.LayerNorm(self.output_dim).cuda()
        self.PositionwiseFeedForward = PositionwiseFeedForward(self.output_dim, self.output_dim*self.layermul)
    
    def ScaleDotProductAttention(self, Q, K, V):
        K_T = K.transpose(-1, -2) # 计算矩阵 K 的转置  
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K_T) / math.sqrt(d_k)
        scores = torch.matmul(Q, K_T) / math.sqrt(d_k)
        attn_weights = self.softmax(scores)
        output = torch.matmul(attn_weights, V)
        return output, attn_weights
    
    def MultiHeadAttention(self, q, k, v):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self.split(q), self.split(k), self.split(v)
        sa_output, attn_weights = self.ScaleDotProductAttention(q, k, v)
        concat_tensor = self.concat(sa_output)
        mha_output = self.fc1(concat_tensor)
        mha_output = mha_output + concat_tensor # 这部分的残差可能只能这么加了
        
        return mha_output
    
    def split(self, tensor):
        batch_size, seq_len, d_model = tensor.size()
        d_tensor = d_model // self.n_head
        split_tensor = tensor.view(batch_size, seq_len, self.n_head, d_tensor).transpose(1, 2)
        return split_tensor
    
    def concat(self, sa_output):
        batch_size, n_head, seq_len, d_tensor = sa_output.size()
        d_model = n_head * d_tensor
        concat_tensor = sa_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        concat_tensor = concat_tensor.view(concat_tensor.shape[0], -1)
        concat_tensor = concat_tensor.view(self.batch, self.neg_sampling, self.output_dim)
        return concat_tensor
    
    def forward(self, obj_embedding):
        self.batch, self.neg_sampling = obj_embedding.shape[0], obj_embedding.shape[1]
        obj_embedding = obj_embedding.view(self.batch * self.neg_sampling, self.input_dim)
        obj_embedding = obj_embedding.view(self.batch * self.neg_sampling, self.seq_len, -1)
        
        obj_embedding = self.MultiHeadAttention(obj_embedding, obj_embedding, obj_embedding)
        obj_embedding = self.LayerNorm1(obj_embedding)
        
        obj_embedding = obj_embedding + self.PositionwiseFeedForward(obj_embedding)
        obj_embedding = self.LayerNorm2(obj_embedding)
        
        return obj_embedding