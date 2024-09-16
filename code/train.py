# long-term using attention
# 使用注意力机制进行长期建模

import argparse
import json
import pandas as pd  # 导入pandas库，用于数据处理
import os  # 导入os库，用于操作系统接口
import numpy as np  # 导入numpy库，用于数值计算
from sklearn import preprocessing  # 从sklearn库中导入预处理模块
from sklearn.model_selection import train_test_split  # 从sklearn库中导入数据分割函数
import torch  # 导入torch库，用于深度学习操作
import torch.nn as nn  # 从torch库中导入神经网络模块
from torch.autograd import Variable  # 导入Variable模块，用于创建可微分的张量
from torch.nn.parameter import Parameter  # 导入Parameter模块，用于定义可学习的参数
import torch.utils.data as Data  # 从torch库中导入数据处理模块
from torch.backends import cudnn  # 从torch库中导入cudnn后端
import torch.nn.functional as F  # 导入功能模块，包含神经网络中的常用函数
import torch.optim as optim  # 从torch库中导入优化器模块
import torch.nn.utils as utils  # 从torch库中导入实用工具模块
# 从torch库中导入RNN数据处理函数
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pickle  # 导入pickle库，用于数据序列化
import sys  # 导入sys库，用于系统相关操作
import codecs  # 导入codecs库，用于编解码操作
from torch.utils.tensorboard import SummaryWriter
import utils as preprocess  # 导入自定义的长短期预处理模块
import model  # 导入自定义的长短期模型
from datetime import datetime

SEED = 0  # 设置随机种子
torch.manual_seed(SEED)  # 为CPU设置随机种子
torch.cuda.manual_seed(SEED)  # 为所有GPU设置随机种子
# 确保每次运行代码时生成的随机数序列相同，这样可以使实验结果具有可重现性，方便调试和比较不同模型的性能。
parser = argparse.ArgumentParser(
    description='Train a model with given configuration.')
parser.add_argument('--config', type=str, required=True,
                    help='Path to the JSON configuration file')
args = parser.parse_args()

# 读取 JSON 文件
with open(args.config, 'r') as f:
    config = json.load(f)

'''
# Training Parameters
# 训练参数设置
batch_size = 32     # !批大小,改这里先改utils的输入batch_size
# 隐藏层大小    128 调参工具：使用如 Hyperopt、Optuna 或 Grid Search 等自动化超参数调优工具，可以帮助系统地探索不同的 hidden_size 配置。
hidden_size = 128
num_layers = 1  # 网络层数  LSTM
num_epochs = 15     # !训练的轮数
lr = 0.0005          # 学习率

#ToDo 学习率调度器
# 定义学习率调度器，每隔10个epoch将学习率降低0.1倍
#scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

vocab_hour = 48     # 小时的词汇大小
vocab_week = 7      # 星期的词汇大小

embed_poi = 300     # 兴趣点的 嵌入维度 ？
embed_cat = 100     # 类别的嵌入维度
embed_user = 50     # 用户的嵌入维度
embed_hour = 20     # 小时的嵌入维度    sccode+user+hour = 270
embed_week = 7      # 星期的嵌入维度
embed_wekn = 52     # 周数的嵌入维度

run_name = "3type_vp9880_"   # 定义运行名称
'''
batch_size = config['batch_size']     # !批大小,改这里先改utils的输入batch_size
# 隐藏层大小    128 调参工具：使用如 Hyperopt、Optuna 或 Grid Search 等自动化超参数调优工具，可以帮助系统地探索不同的 hidden_size 配置。
hidden_size = config['hidden_size']
num_layers = config['num_layers']  # 网络层数  LSTM
num_epochs = config['num_epochs']     # !训练的轮数
lr = config['lr']          # 学习率

# ToDo 学习率调度器
# 定义学习率调度器，每隔10个epoch将学习率降低0.1倍
# scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

vocab_hour = 48     # 小时的词汇大小
vocab_week = 7      # 星期的词汇大小

embed_poi = config['embed_poi']     # 兴趣点的 嵌入维度 ？
embed_cat = config['embed_cat']     # 类别的嵌入维度
embed_user = config['embed_user']     # 用户的嵌入维度
embed_hour = config['embed_hour']     # 小时的嵌入维度    sccode+user+hour = 270
embed_week = config['embed_week']      # 星期的嵌入维度
embed_wekn = config['embed_wekn']     # 周数的嵌入维度

run_name = config['run_name']   # 定义运行名称

current_time_str = datetime.now().strftime('%m-%d %H:%M').replace(' ', '_')
filepathname = run_name + current_time_str+'_'+str(num_epochs) + "ep"
log = open("../../autodl-tmp/LogFiles/log_" +
           filepathname + ".txt", "w")  # 打开一个文件用于记录日志
sys.stdout = log                            # 将标准输出重定向到日志文件
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置CUDA设备可见性

print("emb_poi :", embed_poi)  # 打印兴趣点嵌入维度
print("emb_cat :", embed_cat)
print("emb_user :", embed_user)  # 打印用户嵌入维度
print("emb_hour :", embed_hour)
print("emb_week :", embed_week)
print("emb_wekn :", embed_wekn)
print("batch_size : ", batch_size)
print("epochs_num :", num_epochs)
print("hidden_size :", hidden_size)  # 打印隐藏层大小
print("learning_rate :", lr)  # 打印学习率


print("start preprocess")   # 开始预处理
# pre_data = preprocess.sliding_varlen(data, batch_size) # 调用预处理函数---**********@@@@@@@@@+++++++++++++++++========
print("Data preprocess Done.")
# 预处理完成

# NOTE1.加载预处理后的数据
with open("../data/pre_data.txt", "rb") as f:  # 打开预处理数据文件
    pre_data = pickle.load(f)  # 加载预处理数据
with open("../data/long_term.pk", "rb") as f:  # 打开长期特征数据文件
    long_term = pickle.load(f)  # 加载长期特征数据
with open("../data/cat_candidate.pk", "rb") as f:  # 打开类别候选数据文件
    cat_candi = pickle.load(f)  # 加载类别候选数据
    # TODO 如果表现不好，把这类别候选数据+sccode 不是真的候选！？
with open('../data/datadistance_1type_2.pkl', 'rb') as f:
    poi_distance_matrix = pickle.load(f)  # 加载数据距离矩阵
poi_distance_matrix = torch.tensor(
    poi_distance_matrix).cuda()  # !dtype=torch.float
# 打开venueID和经纬度的映射数据
with open('../data/venueid2coor2seqnumber_1type.pkl', 'rb') as f:
    venueid2coor = pickle.load(f)  # 加载venueID和经纬度的映射数据
with open('../data/venue_freq_matrix.pkl', 'rb') as f:
    venue_freq_matrix = pickle.load(f)  # 加载venue频次矩阵
venue_freq_matrix = torch.tensor(venue_freq_matrix, dtype=torch.float).cuda()
with open('../data/catid_time_matrix.pkl', 'rb') as f:
    catid_time_matrix = pickle.load(f)
catid_time_matrix = torch.tensor(catid_time_matrix, dtype=torch.float).cuda()

# with open('long_term_feature.pk','rb') as f:
# 	long_term_feature = pickle.load(f)                  # 加载长期特征数据
long_term_feature = [0]     # 初始化长期特征为0

# [ ]:".pk文件内容"
cat_candi = torch.cat((torch.Tensor([0]), cat_candi))  # 合并额外的类别到类别候选中
# 转换类别候选数据类型为长整型，可能是为了之后的索引或分类操作，因为在处理类别数据时，整数类型比浮点数类型更常用。
cat_candi = cat_candi.long()

[vocab_poi, vocab_cat, vocab_size_week_n, vocab_user, len_train,
    len_test] = pre_data["size"]  # 从预处理数据中获取尺寸信息

loader_train = pre_data["loader_train"]  # 获取训练数据加载器
loader_test = pre_data["loader_test"]  # 获取测试数据加载器

print("train set size: ", len_train)  # 打印训练集大小
print("test set size: ", len_test)  # 打印测试集大小
print("vocab_poi: ", vocab_poi)  # 打印兴趣点词汇大小
print("vocab_cat: ", vocab_cat)  # 打印类别词汇大小

print("Train the Model...")  # 开始训练模型

# NOTE2. 初始化模型
Model = model.long_short(
    embed_user,
    embed_poi,
    embed_cat,
    embed_wekn,
    embed_hour,
    embed_week,
    hidden_size,
    num_layers,
    vocab_poi + 1,  # ？加 1 的原因通常是因为要留出一个额外的索引位置用于表示未知的用户或者填充的用户。
    vocab_cat + 1,
    vocab_size_week_n+1,
    vocab_user + 1,
    vocab_hour,
    long_term,
    cat_candi,
    venueid2coor,
    poi_distance_matrix,
    catid_time_matrix,
    # len(long_term_feature[0]),
)

userid_cursor = False  # 初始化用户ID游标
results_cursor = False  # 初始化结果游标

Model = Model.cuda()  # 将模型转移到CUDA

# NOTE3.损失函数和优化器
loss_function = nn.CrossEntropyLoss()  # 定义损失函数为交叉熵损失
# optimizer = optim.Adam(Model.parameters(), lr)  # 定义优化器为Adam
# ToDo 替换优化器为AdamW
weight_decay = 0.01  # 权重衰减系数，你可以根据需要调整
optimizer = optim.AdamW(Model.parameters(), lr=lr, weight_decay=weight_decay)

# 优化器 负责根据模型的损失函数调整模型参数

# 平均准确率    预测正确与否，而不考虑正确预测的顺序

# precision = recall

#! =recall


def precision(indices, batch_y, k, count, delta_dist):  # 平均准确率
    precision = 0  # 初始化准确率
    for i in range(indices.size(0)):  # 遍历每一个预测
        sort = indices[i]  # 获取预测排序
        if batch_y[i].long() in sort[:k]:  # 如果真实标签在前k个预测中
            precision += 1  # 准确率增加
    return precision / count  # 返回平均准确率


def precision_a(indices, batch_y, k, count, delta_dist):
    '''### gpt vision'''
    precision = 0
    for i in range(indices.size(0)):
        sort = indices[i]
        correct = 0
        # 检查了在前k个中是否存在。
        for j in range(k):
            if batch_y[i].long().item() == sort[j].item():
                correct += 1
        precision += correct / k
    return precision / count

# 平均精确率 top K  正确预测的顺序，更适合评价排序的质量


def MAP(indices, batch_y, k, count):
    sum_precs = 0  # 初始化平均精确率总和
    for i in range(indices.size(0)):  # 遍历每一个预测
        sort = indices[i]  # 获取预测排序
        ranked_list = sort[:k]  # 获取前k个预测
        hists = 0
        for n in range(len(ranked_list)):  # 遍历前k个预测
            if ranked_list[n].cpu().numpy() in batch_y[i].long().cpu().numpy():  # 如果预测在真实标签中
                hists += 1
                sum_precs += hists / (n + 1)  # 更新累计精确率
    return sum_precs / count  # 返回平均精确率

# 召回率    正样本占所有正样本的比例。（正样本是指在数据集中具有所需属性或标签的样本。


def recall(indices, batch_y, k, count, delta_dist):
    '''计算在前 k 个预测中，是否包含真实标签。'''
    recall_correct = 0
    for i in range(indices.size(0)):
        sort = indices[i]
        if batch_y[i].long().item() in sort[:k].cpu().numpy():
            recall_correct += 1
    return recall_correct / count

# TODO 有待check


def f1_score(precision_k, recall_k):
    if precision_k + recall_k == 0:
        return 0  # 避免分母为0的情况
    f1_score_k = 2 * (precision_k * recall_k) / (precision_k + recall_k)

    return f1_score_k


# TODO Functino DNCG()
# def NDCG() #结合排名的准确性和排名的相关性。NDCG通过比较模型生成的排名列表与真实排名列表之间的相关性来评估模型的性能。
# 衡量推荐结果的相关性和排序质量
# 调整后的 log2 从2开始
def dcg_at_k(relevance_scores, k):
    relevance_scores = np.array(relevance_scores)[:k]
    return np.sum((2**relevance_scores - 1) / np.log2(np.arange(2, len(relevance_scores) + 2)))


def ndcg_at_k(relevance_scores, k):
    actual_dcg = dcg_at_k(relevance_scores, k)
    ideal_relevance_scores = sorted(relevance_scores, reverse=True)
    ideal_dcg = dcg_at_k(ideal_relevance_scores, k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0


def NDCG(indices, batch_y, k):
    ndcg_scores = 0
    for i in range(indices.size(0)):
        relevance_scores = []
        sort = indices[i]
        for rank, predicted in enumerate(sort[:k]):
            if predicted.cpu().numpy() in batch_y[i].long().cpu().numpy():
                relevance_scores.append(1)
            else:
                relevance_scores.append(0)
        ndcg_scores += ndcg_at_k(relevance_scores, k)
    return ndcg_scores / indices.size(0)


writer_path = config['tensorboard_runs/']
writer = SummaryWriter(writer_path)

# NOTE4.迭代所有训练周期
for epoch in range(num_epochs):
    Model = Model.train()  # 设置模型为训练模式
    total_loss = 0.0  # 初始化总损失

    # 初始化各项评估指标
    precision_1 = 0
    precision_5 = 0
    precision_10 = 0
    precision_20 = 0

    precision_a_1 = 0
    precision_a_5 = 0
    precision_a_10 = 0
    precision_a_20 = 0

    recall_1 = 0
    recall_5 = 0
    recall_10 = 0
    recall_20 = 0

    MAP_1 = 0
    MAP_5 = 0
    MAP_10 = 0
    MAP_20 = 0

    ndcg_1 = 0
    ndcg_5 = 0
    ndcg_10 = 0
    ndcg_20 = 0

    f1_score_1 = 0
    f1_score_5 = 0
    f1_score_10 = 0
    f1_score_20 = 0

    userid_wrong_train = {}
    userid_wrong_test = {}
    results_train = []
    results_test = []

    # 从训练数据加载器中获取数据
    for step, (batch_x, batch_x_cat, batch_x_wekn, batch_y, hours, batch_userid, hour_pre, week_pre) in enumerate(loader_train):
        Model.zero_grad()  # 清空之前的梯度
        users = batch_userid.cuda()  # 将用户ID数据移至CUDA
        hourids = Variable(hours.long()).cuda()  # 将小时数据转换为长整型变量并移至CUDA

        # 将所有输入数据转换为CUDA变量
        batch_x, batch_x_cat, batch_x_wekn, batch_y, hour_pre, week_pre = (
            Variable(batch_x).cuda(),
            Variable(batch_x_cat).cuda(),
            Variable(batch_x_wekn).cuda(),
            Variable(batch_y).cuda(),
            Variable(hour_pre.long()).cuda(),
            Variable(week_pre.long()).cuda(),
        )

        # 生成兴趣点候选列表
        poi_candidate = list(range(vocab_poi + 1))
        poi_candi = Variable(torch.LongTensor(poi_candidate)
                             ).cuda()  # 将兴趣点候选转换为长整型变量并移至CUDA
        cat_candi = Variable(cat_candi).cuda()  # 将类别候选数据转换为变量并移至CUDA
        # TODO poi_candi 和 cat_candi没用到？

        outputs = Model(
            batch_x, batch_x_cat, batch_x_wekn, users, hourids, hour_pre, week_pre, poi_candi, cat_candi, venueid2coor, poi_distance_matrix, venue_freq_matrix, catid_time_matrix
        )  # 调用模型进行预测

        loss = 0
        for i in range(batch_x.size(0)):    # 遍历批次中的每一个样本
            # print("outputs[i, :, :]:",outputs[i, :, :])
            # print("batch_y[i, :]: ",batch_y[i, :])
            loss += loss_function(outputs[i, :, :],
                                  batch_y[i, :]).cuda()   # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 优化器更新模型参数
        total_loss += float(loss)   # 累计损失

        outputs2 = outputs[:, -1, :]  # ! 获取输出最后一层的 why
        batch_y2 = batch_y[:, -1]  # batch_y , utils时，train_y是最后一个结果

        out_p, indices = torch.sort(
            outputs2, dim=1, descending=True)  # 对输出进行排序
        count = float(len_train)  # 获取训练数据总数
        delta_dist = 0
        # 计算不同k值的精确率
        precision_1 += precision(indices, batch_y2, 1, count, delta_dist)
        precision_5 += precision(indices, batch_y2, 5, count, delta_dist)
        precision_10 += precision(indices, batch_y2, 10, count, delta_dist)
        precision_20 += precision(indices, batch_y2, 20, count, delta_dist)

        precision_a_1 += precision_a(indices, batch_y2, 1, count, delta_dist)
        precision_a_5 += precision_a(indices, batch_y2, 5, count, delta_dist)
        precision_a_10 += precision_a(indices, batch_y2, 10, count, delta_dist)
        precision_a_20 += precision_a(indices, batch_y2, 20, count, delta_dist)

        recall_1 += recall(indices, batch_y2, 1, count, delta_dist)
        recall_5 += recall(indices, batch_y2, 5, count, delta_dist)
        recall_10 += recall(indices, batch_y2, 10, count, delta_dist)
        recall_20 += recall(indices, batch_y2, 20, count, delta_dist)

        # 计算不同k值的MAP
        MAP_1 += MAP(indices, batch_y2, 1, count)
        MAP_5 += MAP(indices, batch_y2, 5, count)
        MAP_10 += MAP(indices, batch_y2, 10, count)
        MAP_20 += MAP(indices, batch_y2, 20, count)

        # 计算NDCG结果
        ndcg_1 = NDCG(indices, batch_y2, 1)
        ndcg_5 = NDCG(indices, batch_y2, 5)
        ndcg_10 = NDCG(indices, batch_y2, 10)
        ndcg_20 = NDCG(indices, batch_y2, 20)

        # f1_score_1 += f1_score(indices, batch_y2, 1, count, delta_dist)
        f1_score_5 += f1_score(precision_5, recall_5)
        f1_score_10 += f1_score(precision_10, recall_10)
        # f1_score_20 += f1_score(indices, batch_y2, 20, count, delta_dist)

    # 打印训练结果
    print(
        "train:",
        "epoch: [{}/{}]\t".format(epoch, num_epochs),
        "loss: {:.4f}\t".format(total_loss),
        # "precision@1: {:.4f}\t".format(precision_1),
        # "precision@5: {:.4f}\t".format(precision_5),
        # "precision@10: {:.4f}\t".format(precision_10),
        # "precision@20: {:.4f}\t".format(precision_20),
        # "precision_a@1: {:.4f}\t".format(precision_a_1),
        # "precision_a@5: {:.4f}\t".format(precision_a_5),
        # "precision_a@10: {:.4f}\t".format(precision_a_10),
        # "precision_a@20: {:.4f}\t".format(precision_a_20),
        "recall@1: {:.4f}\t".format(recall_1),
        "recall@5: {:.4f}\t".format(recall_5),
        "recall@10: {:.4f}\t".format(recall_10),
        "recall@20: {:.4f}\t".format(recall_20),
        "MAP@1: {:.4f}\t".format(MAP_1),
        "MAP@5: {:.4f}\t".format(MAP_5),
        "MAP@10: {:.4f}\t".format(MAP_10),
        "MAP@20: {:.4f}\t".format(MAP_20),
        "NDCG@1: {:.4f}\t".format(ndcg_1),
        "NDCG@5: {:.4f}\t".format(ndcg_5),
        "NDCG@10: {:.4f}\t".format(ndcg_10),
        "NDCG@20: {:.4f}\t".format(ndcg_20),
        # "F1@1: {:.4f}\t".format(f1_score_1),
        # "F1@5: {:.4f}\t".format(f1_score_5),
        # "F1@10: {:.4f}\t".format(f1_score_10),
        # "F1@20: {:.4f}\t".format(f1_score_20),
    )
    writer.add_scalar('Loss/train', total_loss, epoch)
    # writer.add_scalar('Train_pre/Precision@1', precision_1, epoch)
    # writer.add_scalar('Train_pre/Precision@5', precision_5, epoch)
    # writer.add_scalar('Train_pre/Precision@10', precision_10, epoch)
    # writer.add_scalar('Train_pre/Precision@20', precision_20, epoch)
    # writer.add_scalar('Train_pre/Precision_a@1', precision_a_1, epoch)
    # writer.add_scalar('Train_pre/Precision_a@5', precision_a_5, epoch)
    # writer.add_scalar('Train_pre/Precision_a@10', precision_a_10, epoch)
    # writer.add_scalar('Train_pre/Precision_a@20', precision_a_20, epoch)
    writer.add_scalar('Train_Rec/Recall@1', recall_1, epoch)
    writer.add_scalar('Train_Rec/Recall@5', recall_5, epoch)
    writer.add_scalar('Train_Rec/Recall@10', recall_10, epoch)
    writer.add_scalar('Train_Rec/Recall@20', recall_20, epoch)
    writer.add_scalar('Train_map/MAP@1', MAP_1, epoch)
    writer.add_scalar('Train_map/MAP@5', MAP_5, epoch)
    writer.add_scalar('Train_map/MAP@10', MAP_10, epoch)
    writer.add_scalar('Train_map/MAP@20', MAP_20, epoch)
    # 将NDCG结果写入TensorBoard
    writer.add_scalar('Train_ndcg/NDCG@1', ndcg_1, epoch)
    writer.add_scalar('Train_ndcg/NDCG@5', ndcg_5, epoch)
    writer.add_scalar('Train_ndcg/NDCG@10', ndcg_10, epoch)
    writer.add_scalar('Train_ndcg/NDCG@20', ndcg_20, epoch)
    # 将F1-score结果写入TensorBoard
    # writer.add_scalar('Train_f1/F1@1', f1_score_1, epoch)
    # writer.add_scalar('Train_f1/F1@5', f1_score_5, epoch)
    # writer.add_scalar('Train_f1/F1@10', f1_score_10, epoch)
    # writer.add_scalar('Train_f1/F1@20', f1_score_20, epoch)

    # 检查并创建模型保存目录
    savedir = "../../autodl-tmp/checkpoint_file/checkpoint_" + filepathname
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    # 设置模型保存路径
    savename = savedir + "/checkpoint" + "_" + str(epoch) + ".tar"

    # 保存模型
    torch.save({"epoch": epoch + 1, "state_dict": Model.state_dict(), }, savename)

    if epoch % 1 == 0:  # 每一个epoch后执行验证

        Model = Model.eval()  # 设置模型为评估模式

        total_loss = 0.0  # 初始化总损失

        # 重置各项评估指标
        precision_1 = 0
        precision_5 = 0
        precision_10 = 0
        precision_20 = 0

        precision_a_1 = 0
        precision_a_5 = 0
        precision_a_10 = 0
        precision_a_20 = 0

        recall_1 = 0
        recall_5 = 0
        recall_10 = 0
        recall_20 = 0

        MAP_1 = 0
        MAP_5 = 0
        MAP_10 = 0
        MAP_20 = 0

        ndcg_1 = 0
        ndcg_5 = 0
        ndcg_10 = 0
        ndcg_20 = 0

        f1_score_1 = 0
        f1_score_5 = 0
        f1_score_10 = 0
        f1_score_20 = 0

        num_val = -1
        # 从测试数据加载器中获取数据并进行评估
        for step, (batch_x, batch_x_cat, batch_x_wekn, batch_y, hours, batch_userid, hour_pre, week_pre) in enumerate(loader_test):
            # num_val += 1
            Model.zero_grad()  # 清空梯度
            hourids = hours.long()  # 将小时数据转换为长整型
            users = batch_userid  # 获取用户ID

            # 将所有输入数据转换为CUDA变量
            batch_x, batch_x_cat, batch_x_wekn, batch_y, hour_pre, week_pre = (
                Variable(batch_x).cuda(),
                Variable(batch_x_cat).cuda(),
                Variable(batch_x_wekn).cuda(),
                Variable(batch_y).cuda(),
                Variable(hour_pre.long()).cuda(),
                Variable(week_pre.long()).cuda(),
            )
            users = Variable(users).cuda()  # 将用户ID数据转移到CUDA
            hourids = Variable(hourids).cuda()  # 将小时数据转移到CUDA

            # 调用模型进行预测
            outputs = Model(
                batch_x, batch_x_cat, batch_x_wekn, users, hourids, hour_pre, week_pre, poi_candi, cat_candi, venueid2coor, poi_distance_matrix, venue_freq_matrix, catid_time_matrix)

            loss = 0
            for i in range(batch_x.size(0)):    # 遍历批次中的每一个样本并计算损失
                loss += loss_function(outputs[i, :, :], batch_y[i, :])

            total_loss += float(loss)  # 累计损失

            outputs2 = outputs[:, -1, :]  # 获取输出的最后一层
            batch_y2 = batch_y[:, -1]  # 获取标签的最后一层

            weights_output = outputs2.data  # 获取权重输出

            outputs2 = weights_output  # 更新输出数据
            out_p, indices = torch.sort(
                outputs2, dim=1, descending=True)  # 对输出进行排序

            count = float(len_test)  # 获取测试数据总数

            # 计算不同k值的精确率和MAP
            precision_1 += precision(indices, batch_y2, 1, count, delta_dist)
            precision_5 += precision(indices, batch_y2, 5, count, delta_dist)
            precision_10 += precision(indices, batch_y2, 10, count, delta_dist)
            precision_20 += precision(indices, batch_y2, 20, count, delta_dist)

            precision_a_1 += precision_a(indices,
                                         batch_y2, 1, count, delta_dist)
            precision_a_5 += precision_a(indices,
                                         batch_y2, 5, count, delta_dist)
            precision_a_10 += precision_a(indices,
                                          batch_y2, 10, count, delta_dist)
            precision_a_20 += precision_a(indices,
                                          batch_y2, 20, count, delta_dist)

            recall_1 += recall(indices, batch_y2, 1, count, delta_dist)
            recall_5 += recall(indices, batch_y2, 5, count, delta_dist)
            recall_10 += recall(indices, batch_y2, 10, count, delta_dist)
            recall_20 += recall(indices, batch_y2, 20, count, delta_dist)

            MAP_1 += MAP(indices, batch_y2, 1, count)
            MAP_5 += MAP(indices, batch_y2, 5, count)
            MAP_10 += MAP(indices, batch_y2, 10, count)
            MAP_20 += MAP(indices, batch_y2, 20, count)

            # 计算NDCG结果
            ndcg_1 = NDCG(indices, batch_y2, 1)
            ndcg_5 = NDCG(indices, batch_y2, 5)
            ndcg_10 = NDCG(indices, batch_y2, 10)
            ndcg_20 = NDCG(indices, batch_y2, 20)

            # f1_score_1 += f1_score(indices, batch_y2, 1, count, delta_dist)
            f1_score_5 += f1_score(precision_5, recall_5)
            f1_score_10 += f1_score(precision_5, recall_10)
            # f1_score_20 += f1_score(indices, batch_y2, 20, count, delta_dist)

        # 打印验证结果
        print(
            "val: {}\t".format(epoch),
            "loss: {:.4f}\t".format(total_loss),
            # 精确率。Precision@k，仅考虑排名在前k位的预测结果
            # "precision@1: {:.4f}\t".format(precision_1),
            # "precision@5: {:.4f}\t".format(precision_5),
            # "precision@10: {:.4f}\t".format(precision_10),
            # "precision@20: {:.4f}\t".format(precision_20),
            # "precision_a@1: {:.4f}\t".format(precision_a_1),
            # "precision_a@5: {:.4f}\t".format(precision_a_5),
            # "precision_a@10: {:.4f}\t".format(precision_a_10),
            # "precision_a@20: {:.4f}\t".format(precision_a_20),
            "recall@1: {:.4f}\t".format(recall_1),
            "recall@5: {:.4f}\t".format(recall_5),
            "recall@10: {:.4f}\t".format(recall_10),
            "recall@20: {:.4f}\t".format(recall_20),
            "MAP@1: {:.4f}\t".format(MAP_1),
            "MAP@5: {:.4f}\t".format(MAP_5),
            "MAP@10: {:.4f}\t".format(MAP_10),
            "MAP@20: {:.4f}\t".format(MAP_20),
            "NDCG@1: {:.4f}\t".format(ndcg_1),
            "NDCG@5: {:.4f}\t".format(ndcg_5),
            "NDCG@10: {:.4f}\t".format(ndcg_10),
            "NDCG@20: {:.4f}\t".format(ndcg_20),
            # "F1@1: {:.4f}\t".format(f1_score_1),
            # "F1@5: {:.4f}\t".format(f1_score_5),
            # "F1@10: {:.4f}\t".format(f1_score_10),
            # "F1@20: {:.4f}\t".format(f1_score_20),
        )
        writer.add_scalar('Loss/val', total_loss, epoch)
        # writer.add_scalar('Val_pre/Precision@1', precision_1, epoch)
        # writer.add_scalar('Val_pre/Precision@5', precision_5, epoch)
        # writer.add_scalar('Val_pre/Precision@10', precision_10, epoch)
        # writer.add_scalar('Val_pre/Precision@20', precision_20, epoch)
        # writer.add_scalar('Val_pre/Precision_a@1', precision_a_1, epoch)
        # writer.add_scalar('Val_pre/Precision_a@5', precision_a_5, epoch)
        # writer.add_scalar('Val_pre/Precision_a@10', precision_a_10, epoch)
        # writer.add_scalar('Val_pre/Precision_a@20', precision_a_20, epoch)
        writer.add_scalar('Val_Rec/Recall@1', recall_1, epoch)
        writer.add_scalar('Val_Rec/Recall@5', recall_5, epoch)
        writer.add_scalar('Val_Rec/Recall@10', recall_10, epoch)
        writer.add_scalar('Val_Rec/Recall@20', recall_20, epoch)
        writer.add_scalar('Val_map/MAP@1', MAP_1, epoch)
        writer.add_scalar('Val_map/MAP@5', MAP_5, epoch)
        writer.add_scalar('Val_map/MAP@10', MAP_10, epoch)
        writer.add_scalar('Val_map/MAP@20', MAP_20, epoch)
        # 将NDCG结果写入TensorBoard
        writer.add_scalar('Val_ndcg/NDCG@1', ndcg_1, epoch)
        writer.add_scalar('Val_ndcg/NDCG@5', ndcg_5, epoch)
        writer.add_scalar('Val_ndcg/NDCG@10', ndcg_10, epoch)
        writer.add_scalar('Val_ndcg/NDCG@20', ndcg_20, epoch)
        # 将F1-score结果写入TensorBoard
        # writer.add_scalar('Val_f1/F1@1', f1_score_1, epoch)
        # writer.add_scalar('Val_f1/F1@5', f1_score_5, epoch)
        # writer.add_scalar('Val_f1/F1@10', f1_score_10, epoch)
        # writer.add_scalar('Val_f1/F1@20', f1_score_20, epoch)

    # writer.add_graph(Model, (batch_x, batch_x_cat, batch_x_wekn, users, hourids, hour_pre, week_pre,
    #             poi_candi, cat_candi, venueid2coor, poi_distance_matrix, venue_freq_matrix, catid_time_matrix))
log.close()
writer.close()
