# 从 PyTorch 中导入构建神经网络所需的模块
import torch.nn as nn
import torch
from torch.autograd import Variable  # 导入 Variable，用于创建可微分的张量
from torch.nn.parameter import Parameter  # 导入 Parameter，用于定义可以学习的网络参数
import torch.nn.functional as F  # 导入功能模块，常用于激活函数等网络操作
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence  # 导入用于处理RNN的实用函数

# 导入用于数据处理的库
import pandas as pd
import numpy as np


class Attn_loc_distance(nn.Module):
    """地点距离注意力机制模块"""

    def __init__(self):
        super(Attn_loc_distance, self).__init__()

    def forward(self, venueid2coor, inputs_poi, poi_distance_matrix):
        # print("poi_distance_matrix.shape: ",poi_distance_matrix.shape)  # [6318, 6318]
        # current_loc_tensor = inputs_poi[:, 0]  #! 获取位置数据,这样根本不对啊，
        # print("length of inputs_poi: ", inputs_poi.shape[0])
        all_attn_energies = []
        for i in range(inputs_poi.shape[0]):
            current_loc_tensor = inputs_poi[i]
            current_loc_np = current_loc_tensor.cpu().numpy()  # 将 PyTorch 张量转换为 NumPy 数组
            current_loc = current_loc_np.tolist()  # 将 NumPy 数组转换为 Python 列表
            # 创建位置索引列表
            current_loc_indices = [venueid2coor[loc] for loc in current_loc]
            # 将位置索引列表转换为 PyTorch 张量
            current_loc_indices_tensor = torch.tensor(
                current_loc_indices).cuda()
            # 获取当前所有位置的距离矩阵
            distance_matrix = poi_distance_matrix[current_loc_indices_tensor, :]
            # 将距离矩阵中的 0 替换为一个很大的值
            # max_distance: 9545.986
            distance_matrix[distance_matrix == 0] = 9999999.99
            # 计算注意力能量
            attn_energies = 1 / distance_matrix
            F.softmax(attn_energies, dim=1)
            all_attn_energies.append(attn_energies)

        final_attn_energies = torch.stack(all_attn_energies, dim=0)
        final_attn_energies.squeeze(1)
        # print("final_attn_energies.shape: ",final_attn_energies.shape)  # [32,9,6318]

        # 返回归一化的注意力权重
        return final_attn_energies


class Attn_loc_freq(nn.Module):
    """地点频率注意力机制模块"""

    def __init__(self):
        super(Attn_loc_freq, self).__init__()

    def forward(self, venueid2coor, inputs_wekn, poi_freq_matrix):
        # 基于地点频率的注意力权重计算
        # 进行按行归一化
        # poi_freq_matrix_normalized_row_sum = poi_freq_matrix_cuda.sum(
        #     dim=1, keepdim=True)
        # poi_freq_matrix_normalized = poi_freq_matrix_cuda / \
        #     poi_freq_matrix_normalized_row_sum
        poi_freq_matrix_cuda = F.softmax(poi_freq_matrix, dim=1)

        # print("\ninputs_wekn.shape: ", inputs_wekn.shape)
        all_attn_energies = []
        for i in range(inputs_wekn.shape[0]):
            current_wekn_tensor = inputs_wekn[i]  # 获取位置数据
            current_wekn_np = current_wekn_tensor.cpu().numpy()  # 将 PyTorch 张量转换为 NumPy 数组
            current_wekn = current_wekn_np.tolist()  # 将 NumPy 数组转换为 Python 列表
            # 使用 PyTorch 的索引功能进行批量操作
            current_wekn_tensor_cuda = torch.tensor(current_wekn).cuda()
            attn_energies = poi_freq_matrix_cuda[current_wekn_tensor_cuda, :]
            all_attn_energies.append(attn_energies)

        final_attn_energies = torch.stack(all_attn_energies, dim=0)
        final_attn_energies.squeeze(1)
        return final_attn_energies

# class Self_Attn_cid(nn.Moudle) 实际旅游中其实并不是很明显


class Attn_cat_freq(nn.Module):
    """类别和时间注意力机制模块"""

    def __init__(self):
        super(Attn_cat_freq, self).__init__()

    def forward(self, inputs_hour, catid_time_matrix):
        # 计算基于类别和时间的注意力权重
        # print("inputs_hour.shape: ", inputs_hour.shape)
        catid_time_matrix_cuda = catid_time_matrix
        catid_time_matrix_cuda = F.softmax(catid_time_matrix_cuda, dim=1)
        all_attn_energies = []
        for i in range(inputs_hour.shape[0]):
            current_hour_tensor = inputs_hour[i]  # 获取位置数据
            current_hour_np = current_hour_tensor.cpu().numpy()  # 将 PyTorch 张量转换为 NumPy 数组
            current_hour = current_hour_np.tolist()  # 将 NumPy 数组转换为 Python 列表
            current_hour_tensor_cuda = torch.tensor(current_hour).cuda()
            attn_energies = catid_time_matrix_cuda[current_hour_tensor_cuda, :]
            all_attn_energies.append(attn_energies)
        final_attn_energies = torch.stack(all_attn_energies, dim=0)
        final_attn_energies.squeeze(1)
        return final_attn_energies


# 定义一个神经网络模型类 long_short，继承自 nn.Module


class long_short(nn.Module):
    def __init__(self, embed_size_user, embed_size_poi, embed_size_cat, embed_size_wekn, embed_size_hour, embed_size_week,
                 hidden_size, num_layers, vocab_poi, vocab_cat, vocab_wekn, vocab_user, vocab_hour, long_term, cat_candi, venueid2coor, poi_distance_matrix, catid_time_matrix):

        super(long_short, self).__init__()  # 初始化父类

        # 定义用户、兴趣点、类别、小时、周的嵌入层
        self.embed_user = nn.Embedding(vocab_user, embed_size_user)
        self.embed_poi = nn.Embedding(vocab_poi, embed_size_poi)
        self.embed_cat = nn.Embedding(vocab_cat, embed_size_cat)
        self.embed_hour = nn.Embedding(vocab_hour, embed_size_hour)
        self.embed_week = nn.Embedding(7, embed_size_week)
        self.embed_wekn = nn.Embedding(vocab_wekn, embed_size_wekn)
        # 定义总嵌入层的尺寸
        self.embed_total_size = embed_size_poi + embed_size_cat + \
            embed_size_hour + embed_size_week
        self.embed_attn_total_size = embed_size_poi + \
            embed_size_cat + embed_size_hour + embed_size_wekn
        # print("self.embed_total_size: ", self.embed_total_size)

        self.vocab_poi = vocab_poi
        self.vocab_cat = vocab_cat
        self.vocab_hour = vocab_hour
        self.vocab_week = 7
        self.vocab_wekn = 52
        self.long_term = long_term  # 长期兴趣模型参数

        # 定义输出层权重：定义的位置在 LSTM 层和线性层之后，用于加权组合不同部分的输出结果。
        self.out_w_long = Parameter(torch.Tensor([0.5]).repeat(vocab_user))
        self.out_w_poi = Parameter(torch.Tensor([0.25]).repeat(vocab_user))
        self.out_w_cat = Parameter(torch.Tensor([0.25]).repeat(vocab_user))

        # 定义权重和偏置参数
        self.weight_poi = Parameter(
            torch.ones(embed_size_poi, embed_size_user))
        self.weight_cat = Parameter(
            torch.ones(embed_size_cat, embed_size_user))
        self.weight_time = Parameter(torch.ones(
            embed_size_hour + embed_size_week, embed_size_user))
        self.bias = Parameter(torch.ones(embed_size_user))

        self.activate_func = nn.ReLU()  # 激活函数

        self.num_layers = num_layers  # LSTM层数
        self.hidden_size = hidden_size

        # 定义隐藏层权重，权重初始化为1
        self.weight_hidden_poi = Parameter(torch.ones(self.hidden_size, 1))
        self.weight_hidden_cat = Parameter(torch.ones(self.hidden_size, 1))

        # 计算嵌入大小总和，用于后续设置LSTM层输入大小
        size = embed_size_poi + embed_size_user + embed_size_hour  # 短左 #300
        # print("lstm_size : ", size)
        size2 = embed_size_cat + embed_size_user + embed_size_hour  # 170			#短右

        # 定义LSTM层和线性层：
        # 对兴趣点 (POI) 使用 LSTM 模型处理输入特征
        self.lstm_poi = nn.LSTM(
            size, hidden_size, num_layers, batch_first=True)
        # 对类别使用 LSTM 模型处理输入特征
        # !双向LSTM : bidirectional=True
        # !TODO 这个层置为双向，结果会怎样？
        self.lstm_cat = nn.LSTM(
            size2, hidden_size, num_layers, batch_first=True)
        # 定义attn后的LSTM
        # !双向LSTM : bidirectional=True
        self.rnn = nn.LSTM(self.embed_attn_total_size, self.hidden_size, 1)
        self.dropout = nn.Dropout(0.3)

        # self.attn_user = Attn_user()
        self.attn_loc_distance = Attn_loc_distance()  # 定义地点注意力机制
        self.attn_loc_freq = Attn_loc_freq()  # 定义地点频率注意力机制
        self.attn_cat_freq = Attn_cat_freq()

        # ! 定义注意力机制的权重？

        # 兴趣点 (POI) 的线性层，用于从隐藏层状态输出预测结果
        # !为什么输入维度是hidden_size?
        self.fc_poi = nn.Linear(hidden_size, self.vocab_poi)
        # 类别的线性层，用于从隐藏层状态输出预测结果
        self.fc_cat = nn.Linear(hidden_size, self.vocab_poi)
        # 长期兴趣的线性层
        self.fc_longterm = nn.Linear(self.embed_total_size, self.vocab_poi)

        self.fc_dis_loc_freq = nn.Linear(
            self.vocab_poi, self.vocab_poi)  # TODO拼接后共用一个线性层？
        # self.fc_loc_freq = nn.Linear(self.vocab_poi, self.vocab_poi)
        self.fc_cat_freq = nn.Linear(self.vocab_cat, self.vocab_poi)

        # self.fc_final = nn.Linear(self.hidden_size, self.vocab_poi)
        # self.fc_user = nn.Linear(embed_size_user, self.vocab_poi)
        self.init_weights()  # LSTM的权重初始化

    def init_weights(self):  # LSTM的权重初始化
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

    # 前向传播

    def forward(self, inputs_poi, inputs_cat, inputs_wekn, inputs_user, inputs_time, hour_pre, week_pre, poi_candi, cat_candi, venueid2coor, poi_distance_matrix, poi_freq_matrix, catid_time_matrix):
        # print("Excuting out_poi:")
        out_poi = self.get_output(inputs_poi, inputs_user, inputs_time, self.embed_poi,
                                  self.embed_user, self.embed_hour, self.lstm_poi, self.fc_poi)
        # print("Excuting out_cat:")
        out_cat = self.get_output(inputs_cat, inputs_user, inputs_time, self.embed_cat,
                                  self.embed_user, self.embed_hour, self.lstm_cat, self.fc_cat)

#########################################################################################

        # 长期兴趣的输出
        u_long = self.get_u_long(inputs_user)
        out_long = self.fc_longterm(u_long).unsqueeze(
            1).repeat(1, out_poi.size(1), 1)

        # 计算加权总输出
        # weighted sum directly
        weight_poi = self.out_w_poi[inputs_user]
        weight_cat = self.out_w_cat[inputs_user]
        # weight_long = self.out_w_long[inputs_user] #32
        weight_long = 1-weight_poi-weight_cat  # 根据用户输入获取兴趣点 (POI) 和类别的权重

        # 应用权重，并扩展维度以匹配输出格式
        out_poi = torch.mul(out_poi, weight_poi.unsqueeze(
            1).repeat(1, 9).unsqueeze(2))  # 乘法运算函数，用于对两个张量进行逐元素相乘。将各部分的输出结果与相应的权重进行逐元素相乘，实现加权操作 [32, 9, 6318]x32
        out_cat = torch.mul(out_cat, weight_cat.unsqueeze(
            1).repeat(1, 9).unsqueeze(2))
        out_long = torch.mul(out_long, weight_long.unsqueeze(
            1).repeat(1, 9).unsqueeze(2))

        # 将各部分加权输出相加，得到最终模型输出 [32, 9, 6318]
        out1 = out_poi + out_cat + out_long

    ###################################################################

        attn_weights_distance = self.attn_loc_distance(
            venueid2coor, inputs_poi, poi_distance_matrix)
        # 地点访问频率的特征权重计算
        attn_weights_loc_freq = self.attn_loc_freq(
            venueid2coor, inputs_wekn, poi_freq_matrix)
        # 类别时间频率的特征权重计算
        attn_weights_cat_freq = self.attn_cat_freq(
            inputs_time, catid_time_matrix)

        poi_emb = self.embed_poi(inputs_poi)
        hour_emb = self.embed_hour(inputs_time)
        wekn_emb = self.embed_wekn(inputs_wekn)
        cat_emb = self.embed_cat(inputs_cat)
        user_emb = self.embed_user(inputs_user).unsqueeze(1).repeat(1, 9, 1)

        #!再加一个Linear？有必要吗，加与不加的效果？
        # print("attn_weights_distance.dtype: ",attn_weights_distance.dtype)
        # TODO 优化这儿代码
        batch_size, seq_len, features = attn_weights_distance.size()
        attn_weights_distance_reshaped = attn_weights_distance.view(
            batch_size * seq_len, features)
        attn_weights_distance_reshaped = attn_weights_distance_reshaped.to(
            self.fc_dis_loc_freq.weight.dtype)
        attn_distance_output = self.fc_dis_loc_freq(
            attn_weights_distance_reshaped)
        attn_distance_output = attn_distance_output.view(
            batch_size, seq_len, features)

        # print("attn_dis_out[1,: ,:]: ",attn_distance_output[1,: ,:])
        attn_distance_output = F.softmax(attn_distance_output, dim=1)
        # print("attn_dis_out_F[1,: ,:]: ",attn_distance_output[1,: ,:])

        #!attn_distance_output = self.fc_distance(attn_weights_distance)
        # print("attn_distance_output:", attn_distance_output.shape)

        attn_loc_freq_output = self.fc_dis_loc_freq(attn_weights_loc_freq)
        # print("attn_loc_freq_out[2, :,:]:",attn_loc_freq_output[2, :, :])
        attn_loc_freq_output = F.softmax(attn_loc_freq_output, dim=1)
        # print("attn_loc_freq_out[2, :,:]: ",attn_loc_freq_output[2, :, :])

        attn_cat_freq_output = self.fc_cat_freq(attn_weights_cat_freq)
        # print("attn_cat_freq_out[2, :,:]: ",attn_cat_freq_output[2, :, :])
        attn_cat_freq_output = F.softmax(attn_cat_freq_output, dim=1)
        # print("attn_cat_freq_out[2, :,:]: ",attn_cat_freq_output[2, :, :])
        # user_output = self.fc_user(user_emb)

        out2 = attn_distance_output+attn_loc_freq_output+attn_cat_freq_output
        #!+user_output?
        # out2 = self.dropout(out2)
        # out2 = self.fc_final(out2)  #TODO再加一个Linear?

        score = out1 + out2

        return score

    # 定义获取用户长期偏好的函数
    # TODO 这个函数还是不太理解！
    def get_u_long(self, inputs_user):
        # get the hidden vector of users' long-term preference
        # 获取用户的长期偏好隐藏向量
        u_long = {}
        for user in inputs_user:
            user_index = user.tolist()
            if user_index not in u_long.keys():

                poi = self.long_term[user_index]['loc']
                hour = self.long_term[user_index]['hour']
                week = self.long_term[user_index]['week']
                cat = self.long_term[user_index]['category']
                # sccode = self.long_term[user_index]['sccode']

                seq_poi = self.embed_poi(poi)  # 转换为嵌入向量
                seq_cat = self.embed_cat(cat)
                # seq_sccode = self.embed_sccode(sccode)  # ! sccode?
                seq_user = self.embed_user(user)
                seq_hour = self.embed_hour(hour)
                seq_week = self.embed_week(week)
                seq_time = torch.cat((seq_hour, seq_week), 1)  # ?

                poi_mm = torch.mm(seq_poi, self.weight_poi)  # 对兴趣点嵌入向量应用线性变换
                cat_mm = torch.mm(seq_cat, self.weight_cat)
                # sccode_mm = torch.mm(seq_sccode, self.weight_sccode)
                time_mm = torch.mm(seq_time, self.weight_time)

                # 将上述所有变换的结果相加并添加偏置
                hidden_vec = poi_mm.add_(cat_mm).add_(time_mm).add_(self.bias)
                hidden_vec = self.activate_func(hidden_vec)  # 876*50	# 激活

                # 计算注意力权重，衡量各项长期偏好的重要性
                alpha = F.softmax(
                    torch.mm(hidden_vec, seq_user.unsqueeze(1)), 0)  # 876*1

                # 将所有嵌入向量合并为一个向量，用于后续的加权求和
                poi_concat = torch.cat(
                    (seq_poi, seq_cat, seq_hour, seq_week), 1)  # 876*427

                # 使用注意力权重 alpha 对合并的向量 poi_concat 进行加权求和
                u_long[user_index] = torch.sum(
                    torch.mul(poi_concat, alpha), 0)  # 计算加权长期偏好
        # 初始化存储所有用户长期偏好向量的张量
        u_long_all = torch.zeros(
            len(inputs_user), self.embed_total_size).cuda()
        # 64*427
        for i in range(len(inputs_user)):  # 根据用户索引，填充每个用户的长期偏好向量
            u_long_all[i, :] = u_long[inputs_user.tolist()[i]]

        return u_long_all

    # 定义获取输出的辅助函数
    def get_output(self, inputs, inputs_user, inputs_time, embed_layer, embed_user, embed_time, lstm_layer, fc_layer):
        # embed your sequences
        seq_tensor = embed_layer(inputs)  # [32,9]
        # 处理用户数据并调整形状以匹配输入序列
        seq_user = embed_user(inputs_user).unsqueeze(1).repeat(
            1, seq_tensor.size(1), 1)  # 32
        seq_time = embed_time(inputs_time)  # 将时间数据通过嵌入层处理

        # print("seq_tensor shape:", seq_tensor.shape)
        # print("seq_user shape:", seq_user.shape)
        # print("seq_time shape:", seq_time.shape)

        # embed your sequences
        input_tensor = torch.cat(
            (seq_tensor, seq_user, seq_time), 2)  # !将各输入数据合并为一个张量

        # pack them up nicely
        output, _ = lstm_layer(input_tensor)  # 通过 LSTM 层处理合并后的输入张量
        out = fc_layer(output)  # the last outputs	# 使用线性层处理 LSTM 的输出，获取最终输出

        return out
