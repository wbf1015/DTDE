import torch
import torch.nn as nn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, input_dim, d_ff=1):
        super(PoswiseFeedForwardNet, self).__init__()
        self.input_dim = input_dim
        self.d_ff = d_ff
        self.fc = nn.Sequential(
            nn.Linear(input_dim, d_ff*input_dim, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff*input_dim, input_dim, bias=False))
        
    def forward(self, inputs):                             # inputs: [batch_size, seq_len, d_model]
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.input_dim).cuda()(output + residual)   # [batch_size, seq_len, d_model]  

class LowDimGenerate(nn.Module):
    def __init__(self, hidden_dim, input_dim, output_dim, d_ff=1):
        super(LowDimGenerate, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_ff = d_ff
        
        self.Basic_Position = nn.Sequential(
            nn.Linear(input_dim, input_dim//4),
            nn.Linear(input_dim//4, output_dim)
        )
        
        self.FT1 = nn.Linear(hidden_dim+output_dim, output_dim//2)
        self.FT2 = nn.Linear(hidden_dim+output_dim, output_dim//2)
        self.FTALL = nn.Linear(hidden_dim+output_dim, output_dim)
        
        self.fc = PoswiseFeedForwardNet(output_dim, d_ff=d_ff)
    
    def forward(self, sem, origin_embedding):
        basic_position = self.Basic_Position(origin_embedding)
        FT1 = self.FT1(torch.cat((sem, basic_position), dim=-1))
        FT2 = self.FT2(torch.cat((sem, basic_position), dim=-1))
        ft_position = basic_position + torch.cat((FT1, FT2), dim=-1)
        
        FTALL = self.FTALL(torch.cat((sem, ft_position), dim=-1))
        ft_position = ft_position + FTALL
        
        outputs = self.fc(ft_position)
        
        return outputs


class RNN(nn.Module):
    def __init__(self, input_dim, position_dim, hidden_dim, output_dim, num_layers, seq_len, bidirectional=True, args=None):
        super(RNN, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.position_dim = position_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.seq_len = seq_len
        
        self.position_embeddings = nn.ParameterList([nn.Parameter(torch.randn(position_dim)) for _ in range(seq_len)])
        
        self.rnn = nn.RNN(input_size=self.input_dim+self.position_dim, hidden_size=self.hidden_dim, batch_first=True, num_layers=self.num_layers,bidirectional=self.bidirectional)
    
    def forward(self, x):
        """输入应该是： x.shape=[batch_size, seq_len, input_dim]"""
        batch_size, seq_len, embedding_dim = x.size()
        
        # 创建一个位置编码张量，形状为 [batch, seq_len, position_dim]
        position_embeddings = position_embeddings = torch.stack([param for param in self.position_embeddings])  # 先得到 [seq_len, position_dim]
        position_embeddings = position_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # 扩展为 [batch, seq_len, position_dim
        # 将 x 和位置编码沿着最后一个维度拼接
        x = torch.cat([x, position_embeddings], dim=-1)  # 最终 x 的形状为 [batch, seq_len, embedding_dim + position_dim]
        
        if self.bidirectional:
            h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(x.device)
        else:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)

        output, _ = self.rnn(x, h0)
        output = output[:,-1,:]
        
        return output


"""一遍RNN提取语义，提取完了就完了"""
class RNNExtracter(nn.Module):
    def __init__(self, args, mode=1):
        super(RNNExtracter, self).__init__()
        self.args = args
        self.mode = mode

        self.position_dim = args.position_dim
        self.num_layers = args.layer
        self.bidirectional = args.bidirectional
        self.seq_len = args.seq_len
        self.num_directions = 2 if self.bidirectional else 1
        
        """mode:1 提取实体的语义信息"""
        if mode==1:
            self.input_dim = (args.input_dim*args.entity_mul)//self.seq_len
            self.hidden_dim = args.hidden_dim
            self.output_dim = args.target_dim*args.entity_mul
        
        """mode:2 提取关系的语义信息"""
        if mode==2:
            self.input_dim = (args.input_dim*args.relation_mul)//self.seq_len
            self.hidden_dim = args.hidden_dim
            self.output_dim = args.target_dim*args.relation_mul
        
        """mode:3 综合实体和关系提取语义信息"""
        if mode==3:
            self.input_dim = (args.input_dim*(args.relation_mul+args.entity_mul))//self.seq_len
            self.hidden_dim = args.hidden_dim
            self.output_dim = args.target_dim*args.entity_mul
        
        
        self.RNN = RNN(self.input_dim, self.position_dim, self.hidden_dim, self.output_dim, self.num_layers, self.seq_len, bidirectional=self.bidirectional, args=None)
        self.fc = nn.Linear(self.hidden_dim*self.num_directions, self.hidden_dim)
        self.LDG = LowDimGenerate(hidden_dim=self.hidden_dim, input_dim=self.input_dim*self.seq_len, output_dim=self.output_dim, d_ff=args.t_dff)
        
    def forward(self, inputs, relation_inputs=None):
        if (self.mode==1) or (self.mode==2):
            batch_size, neg_sampling, embedding_dim = inputs.size()
            inputs_reshaped = inputs.view(batch_size * neg_sampling, self.seq_len, self.input_dim)
            rnn_output = self.RNN(inputs_reshaped)
            output = rnn_output.view(batch_size, neg_sampling, self.num_directions*self.hidden_dim)

            output = self.fc(output)
            
            output = self.LDG(output, inputs)
            
            return output
        
        if self.mode==3:
            batch_size, neg_sampling, embedding_dim1 = inputs.size()
            _, _, embedding_dim2 = relation_inputs.size()
            relation_inputs_expanded = relation_inputs.expand(batch_size, neg_sampling, embedding_dim2)
            combined_inputs = torch.cat((inputs, relation_inputs_expanded), dim=2)
            
            inputs_reshaped = combined_inputs.view(batch_size * neg_sampling, self.seq_len, self.input_dim)
            rnn_output = self.RNN(inputs_reshaped)
            output = rnn_output.view(batch_size, neg_sampling, self.num_directions*self.hidden_dim)

            output = self.fc(output)
            output = self.LDG(output, inputs)
            
            return output