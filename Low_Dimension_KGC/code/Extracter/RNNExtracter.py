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


class CustomRNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len):
        super(CustomRNNCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        
        # 为每个时间步（token）初始化独立的权重和偏置
        self.W_ih = nn.ParameterList([nn.Parameter(torch.randn(input_dim, hidden_dim)) for _ in range(seq_len)])
        self.W_hh = nn.ParameterList([nn.Parameter(torch.randn(hidden_dim, hidden_dim)) for _ in range(seq_len)])
        self.b_h = nn.ParameterList([nn.Parameter(torch.randn(hidden_dim)) for _ in range(seq_len)])

    def forward(self, x, h, t):
        # 计算隐藏状态: h_t = tanh(W_ih[t] * x_t + W_hh[t] * h_t-1 + b_h[t])
        h_next = torch.tanh(x @ self.W_ih[t] + h @ self.W_hh[t] + self.b_h[t])
        return h_next

class CustomRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, seq_len, bidirectional=True, args=None):
        super(CustomRNN, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.seq_len = seq_len
        self.num_directions = 2 if bidirectional else 1
        
        # 创建多层 RNN，每一层有一个前向和一个后向 (如果是双向)
        self.rnn_cells = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.rnn_cells.append(self._create_rnn_layer(input_dim, hidden_dim, seq_len))
            else:
                self.rnn_cells.append(self._create_rnn_layer(hidden_dim, hidden_dim, seq_len))
        
        # 输出层
        # self.fc = nn.Linear(hidden_dim * self.num_directions, output_dim) #在没有低维嵌入生成的时候使用这个
        self.fc = nn.Linear(hidden_dim * self.num_directions, hidden_dim) # 在有低维嵌入生成的时候使用这个

    def _create_rnn_layer(self, input_dim, hidden_dim, seq_len):
        """ 创建一个前向和一个后向的 RNN Cell，如果是双向的 """
        layer = nn.ModuleDict({
            'fwd': CustomRNNCell(input_dim, hidden_dim, seq_len),
        })
        if self.bidirectional:
            layer['bwd'] = CustomRNNCell(input_dim, hidden_dim, seq_len)
        return layer

    def forward(self, x):
        """输入应该是： x.shape=[batch_size, seq_len, input_dim]"""
        batch_size, seq_len, embedding_dim = x.size()
        
        # 初始化隐藏状态，使用全零的初始向量进行初始化
        h = [torch.zeros(batch_size, self.hidden_dim).to(x.device) for _ in range(self.num_layers)]
        if self.bidirectional:
            h_bi = [torch.zeros(batch_size, self.hidden_dim).to(x.device) for _ in range(self.num_layers)]
        
        outputs = []
        outputs_bi = []  # 用来保存反向传播的输出
        
        # 正向传播
        for t in range(seq_len):
            input_fwd = x[:, t, :]  # 提取第 t 个时间步的输入
            for layer in range(self.num_layers):
                h[layer] = self.rnn_cells[layer]['fwd'](input_fwd, h[layer], t)
                
                # 在正向传播时，我们暂时不做反向处理，这里只存储正向的输出
                input_fwd = h[layer]
            
            outputs.append(input_fwd)
        
        # 反向传播
        if self.bidirectional:
            for t in reversed(range(seq_len)):  # 倒序遍历时间步
                input_bwd = x[:, t, :]  # 提取第 t 个时间步的输入
                for layer in range(self.num_layers):
                    h_bi[layer] = self.rnn_cells[layer]['bwd'](input_bwd, h_bi[layer], t)
                    input_bwd = h_bi[layer]
                
                outputs_bi.append(input_bwd)

        # 把时间步的输出堆叠在一起
        outputs = torch.stack(outputs, dim=1)  # 正向的输出
        if self.bidirectional:
            outputs_bi = torch.stack(outputs_bi[::-1], dim=1)  # 反向输出顺序逆转后拼接
            
            # 将正向和反向的输出拼接在一起
            outputs = torch.cat((outputs, outputs_bi), dim=2)  # 拼接正向和反向输出
        
        # 使用最后一个时间步的隐藏状态做分类
        out = self.fc(outputs[:, -1, :])
        
        return out


"""一遍RNN提取语义，提取完了就完了"""
class RNNExtracter(nn.Module):
    def __init__(self, args, mode=1):
        super(RNNExtracter, self).__init__()
        self.args = args
        self.mode = mode

        self.num_layers = args.layer
        self.bidirectional = args.bidirectional
        self.seq_len = args.seq_len
        
        # 生成可训练的参数
        self.entity_embedding = nn.Parameter(
            torch.randn(self.args.input_dim * self.args.entity_mul)
        )
        self.relation_embedding = nn.Parameter(
            torch.randn(self.args.input_dim * self.args.relation_mul)
        )
        
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
        if mode==3 or mode==4:
            self.input_dim = (args.input_dim*(args.entity_mul) + args.input_dim*args.relation_mul)//self.seq_len
            self.hidden_dim = args.hidden_dim
            self.output_dim = args.target_dim
        
        self.RNN = CustomRNN(self.input_dim, self.hidden_dim, self.output_dim, self.num_layers, self.seq_len, bidirectional=self.bidirectional, args=args)
        # 低维嵌入生成
        self.LDG = LowDimGenerate(hidden_dim=self.hidden_dim, input_dim=self.input_dim*self.seq_len, output_dim=self.output_dim, d_ff=args.t_dff)
    
    def forward(self, inputs, relation_inputs=None):
        if (self.mode==1) or (self.mode==2):
            batch_size, neg_sampling, embedding_dim = inputs.size()
            inputs_reshaped = inputs.view(batch_size * neg_sampling, self.seq_len, self.input_dim)
            rnn_output = self.RNN(inputs_reshaped)
            output = rnn_output.view(batch_size, neg_sampling, self.hidden_dim)

            output = self.LDG(output, inputs)
            
            return output
        
        if self.mode==3:
            batch_size, neg_sampling, embedding_dim1 = inputs.size()
            _, _, embedding_dim2 = relation_inputs.size()
            relation_inputs_expanded = relation_inputs.expand(batch_size, neg_sampling, embedding_dim2)
            combined_inputs = torch.cat((inputs, relation_inputs_expanded), dim=2)
            inputs_reshaped = combined_inputs.view(batch_size * neg_sampling, self.seq_len, self.input_dim)

            rnn_output = self.RNN(inputs_reshaped)
            output = rnn_output.view(batch_size, neg_sampling, self.hidden_dim)
            output = self.LDG(output, combined_inputs)

            return output

        if self.mode==4:
            # 获取inputs的维度信息
            batch_size, neg_sampling, embedding_dim1 = inputs.size()
            _, _, embedding_dim2 = relation_inputs.size()

            # 扩展 entity_embedding 以匹配 inputs 的维度
            entity_embedding_expanded = self.entity_embedding.expand(batch_size, neg_sampling, embedding_dim1)

            inputs_with_entity = inputs + entity_embedding_expanded  # 将 entity_embedding 加到 inputs 上

            # 扩展 relation_embedding 以匹配 relation_inputs 的维度
            relation_embedding_expanded = self.relation_embedding.expand(batch_size, neg_sampling, embedding_dim2)
            relation_inputs_with_relation = relation_inputs + relation_embedding_expanded  # 将 relation_embedding 加到 relation_inputs 上

            # 接下来是原来的步骤
            relation_inputs_expanded = relation_inputs_with_relation.expand(batch_size, neg_sampling, embedding_dim2)

            # 将 inputs 和 relation_inputs 连接
            combined_inputs = torch.cat((inputs_with_entity, relation_inputs_expanded), dim=2)

            # Reshape combined_inputs 以输入 RNN
            inputs_reshaped = combined_inputs.view(batch_size * neg_sampling, self.seq_len, self.input_dim)

            # RNN 前向传播
            rnn_output = self.RNN(inputs_reshaped)

            # 将输出 reshape 回原始的 batch_size 和 neg_sampling
            output = rnn_output.view(batch_size, neg_sampling, self.hidden_dim)

            # 使用 LDG 进一步处理输出
            output = self.LDG(output, combined_inputs)

            return output
            