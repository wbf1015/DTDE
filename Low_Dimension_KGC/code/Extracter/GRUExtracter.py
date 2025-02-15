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


class CustomGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CustomGRUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 重置门参数
        self.W_ir = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.W_hr = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.b_r = nn.Parameter(torch.randn(hidden_dim))

        # 更新门参数
        self.W_iz = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.W_hz = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.b_z = nn.Parameter(torch.randn(hidden_dim))

        # 新记忆内容参数
        self.W_in = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.W_hn = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.b_n = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, x, h):
        # 计算重置门: r_t = sigmoid(W_ir * x_t + W_hr * h_t-1 + b_r)
        r_t = torch.sigmoid(x @ self.W_ir + h @ self.W_hr + self.b_r)

        # 计算更新门: z_t = sigmoid(W_iz * x_t + W_hz * h_t-1 + b_z)
        z_t = torch.sigmoid(x @ self.W_iz + h @ self.W_hz + self.b_z)

        # 计算新记忆内容: n_t = tanh(W_in * x_t + r_t * (W_hn * h_t-1) + b_n)
        n_t = torch.tanh(x @ self.W_in + r_t * (h @ self.W_hn) + self.b_n)

        # 计算最终的隐藏状态: h_t = (1 - z_t) * n_t + z_t * h_t-1
        h_next = (1 - z_t) * n_t + z_t * h

        return h_next

class CustomGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, seq_len, bidirectional=True, args=None):
        super(CustomGRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.seq_len = seq_len
        self.num_directions = 2 if bidirectional else 1

        # 创建多层 GRU，每一层有一个前向和一个后向 (如果是双向)
        self.gru_cells = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.gru_cells.append(self._create_gru_layer(input_dim, hidden_dim))
            else:
                self.gru_cells.append(self._create_gru_layer(hidden_dim, hidden_dim))

        # 输出层
        # self.fc = nn.Linear(hidden_dim * self.num_directions, output_dim) #在没有低维嵌入生成的时候使用这个
        self.fc = nn.Linear(hidden_dim * self.num_directions, hidden_dim) # 在有低维嵌入生成的时候使用这个

    def _create_gru_layer(self, input_dim, hidden_dim):
        """ 创建一个前向和一个后向的 GRU Cell，如果是双向的 """
        layer = nn.ModuleDict({
            'fwd': CustomGRUCell(input_dim, hidden_dim),
        })
        if self.bidirectional:
            layer['bwd'] = CustomGRUCell(input_dim, hidden_dim)
        return layer

    def forward(self, x):
        """输入应该是： x.shape=[batch_size, seq_len, input_dim]"""
        batch_size, seq_len, _ = x.size()

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
                h[layer] = self.gru_cells[layer]['fwd'](input_fwd, h[layer])
                input_fwd = h[layer]
            outputs.append(input_fwd)

        # 反向传播
        if self.bidirectional:
            for t in reversed(range(seq_len)):  # 倒序遍历时间步
                input_bwd = x[:, t, :]  # 提取第 t 个时间步的输入
                for layer in range(self.num_layers):
                    h_bi[layer] = self.gru_cells[layer]['bwd'](input_bwd, h_bi[layer])
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
class GRUExtracter(nn.Module):
    def __init__(self, args, mode=1):
        super(GRUExtracter, self).__init__()
        self.args = args
        self.mode = mode

        self.num_layers = args.layer
        self.bidirectional = args.bidirectional
        self.seq_len = args.seq_len
        
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
        
        self.GRU = CustomGRU(self.input_dim, self.hidden_dim, self.output_dim, self.num_layers, self.seq_len, bidirectional=self.bidirectional, args=args)
        # 低维嵌入生成
        self.LDG = LowDimGenerate(hidden_dim=self.hidden_dim, input_dim=self.input_dim*self.seq_len, output_dim=self.output_dim, d_ff=args.t_dff)
    
    def forward(self, inputs, relation_inputs=None):
        if (self.mode==1) or (self.mode==2):
            batch_size, neg_sampling, embedding_dim = inputs.size()
            inputs_reshaped = inputs.view(batch_size * neg_sampling, self.seq_len, self.input_dim)
            rnn_output = self.GRU(inputs_reshaped)
            output = rnn_output.view(batch_size, neg_sampling, self.hidden_dim)

            output = self.LDG(output, inputs)
            
            return output
        
        if self.mode==3:
            batch_size, neg_sampling, embedding_dim1 = inputs.size()
            _, _, embedding_dim2 = relation_inputs.size()
            relation_inputs_expanded = relation_inputs.expand(batch_size, neg_sampling, embedding_dim2)
            combined_inputs = torch.cat((inputs, relation_inputs_expanded), dim=2)
            
            inputs_reshaped = combined_inputs.view(batch_size * neg_sampling, self.seq_len, self.input_dim)
            rnn_output = self.GRU(inputs_reshaped)
            output = rnn_output.view(batch_size, neg_sampling, self.hidden_dim)

            output = self.LDG(output, combined_inputs)

            return output            

if __name__ == '__main__':
    # 模型参数
    input_dim = 10
    hidden_dim = 20
    output_dim = 5
    num_layers = 2
    seq_len = 7
    bidirectional = True

    # 创建模型
    model = CustomGRU(input_dim, hidden_dim, output_dim, num_layers, seq_len, bidirectional, None)

    # 输入张量，形状为 [batch_size=3, seq_len=7, input_dim=10]
    x = torch.randn(3, seq_len, input_dim)

    # 前向传播
    output = model(x)
    print(output.shape)  # 输出形状应该是 [batch_size, output_dim]

