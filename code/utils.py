import pandas as pd  # 导入 pandas 库用于数据处理
import numpy as np  # 导入 numpy 库用于数值计算
import torch  # 导入 PyTorch 库用于构建深度学习模型
import torch.nn.functional as F  # 导入 PyTorch 中的功能模块，主要包含神经网络层
import torch.utils.data as Data  # 导入 PyTorch 的数据处理模块
from sklearn.model_selection import train_test_split  # 从 scikit-learn 中导入用于数据分割的函数
from sklearn.preprocessing import MinMaxScaler  # 从 scikit-learn 中导入数据标准化模块
import datetime  # 导入 datetime 库处理日期和时间
import pickle  # 导入 pickle 库用于数据序列化
import time  # 导入 time 库用于处理时间相关的任务
import os  # 导入 os 库用于操作系统级别的接口，如读写文件
import torch.nn as nn
import torch
from datetime import datetime
import pickle
from collections import defaultdict
from math import radians, sin, cos, asin, sqrt
from tqdm import tqdm
import pandas as pd
from collections import OrderedDict

# 定义处理变长数据的函数


def sliding_varlen(data, batch_size):
    print("function: sliding_varlen() is running...")

    def timedelta(time1, time2):  # 定义计算时间差的函数
        t1 = datetime.datetime.strptime(str(time1), '%a %b %d %H:%M:%S %z %Y')
        t2 = datetime.datetime.strptime(str(time2), '%a %b %d %H:%M:%S %z %Y')
        delta = t1-t2
        time_delta = datetime.timedelta(
            days=delta.days, seconds=delta.seconds).total_seconds()
        return time_delta/3600  # 将时间差转换为小时并返回

    def get_entropy(x):  # 定义计算熵的函数
        x_value_list = set([x[i] for i in range(x.shape[0])])
        ent = 0.0
        for x_value in x_value_list:
            p = float(x[x == x_value].shape[0]) / x.shape[0]
            logp = np.log2(p)
            ent -= p * logp
        return ent

#################################################################################
    # 1、sort the raw data in chronological order
    # NOTE1.
    timestamp = []
    hour = []
    day = []
    week = []
    week_n_list = []

    #TODO 按userid和timesstamp再排序

    for i in range(len(data)):
        times = data['timestamp'].values[i]
        # Correct format string
        t = datetime.strptime(times, '%Y-%m-%d %H:%M:%S%z')
        year = int(t.strftime('%Y'))
        day_i = int(t.strftime('%j'))
        week_i = t.weekday()
        hour_i = int(t.strftime('%H'))
        week_n = t.isocalendar()[1]  # 1-52

        # Adjust hour for weekends
        if week_i == 0 or week_i == 6:
            hour_i = hour_i + 24

        # Adjust day for leap years
        if year == 2013:
            day_i = day_i + 366

        # Append extracted values to corresponding lists
        timestamp.append(time.mktime(t.timetuple()))
        day.append(day_i)
        hour.append(hour_i)
        week.append(week_i)
        week_n_list.append(week_n)

    data['timestamp'] = timestamp  # 添加时间戳
    data['hour'] = hour  # 添加小时信息
    print(data['hour'][10:])
    data['day'] = day  # 添加天信息
    data['week'] = week  # 添加周信息
    data['week_n_list'] = week_n_list
#! 嵌入周，和周预测等等与周相关的 TODO
    # TODO 看一下PG2Net的时间划分片是怎么来的

    # data.sort_values(by = 'timestamp',inplace=True,ascending = True)	# 将数据按时间戳升序排序
    # TODO: 这里既然不排序了，那就输出一下看看，是否符合预期

#################################################################################
    # 2、filter users and POIs
    # NOTE2.筛选用户和地点（POI）
    data['userid'] = data['userid'].rank(method='dense').values  # 将用户ID转换为密集排名
    data['userid'] = data['userid'].astype(int)
    data['venueid'] = data['venueid'].rank(
        method='dense').values  # !将地点ID转换为密集排名 从1开始，venueid相同则排名相同
    data['venueid'] = data['venueid'].astype(int)
    for venueid, group in data.groupby('venueid'):
        indexs = group.index
        if len(set(group['catid'].values)) > 1:  # 如果同一地点有多个类别ID
            for i in range(len(group)):
                data.loc[indexs[i], 'catid'] = group.loc[indexs[0]
                                                         ]['catid']  # 统一类别ID

    data = data.drop_duplicates()  # 删除重复项
    data['catid'] = data['catid'].rank(method='dense').values  # 将类别ID转换为密集排名
    data['catid'] = data['catid'].astype(int)
# data['catid'] = data['catid'].rank(method='dense').values.astype(int)  # 将类别ID转换为密集排名

    # data['sccode'] = data['sccode'].rank(method='dense').values
    # data['sccode'] = data['sccode'].astype(int)

    # data['week_n_list'] 无须rank, 1-52

#################################################################################
    poi_cat = data[['venueid', 'catid']]  # TODO 要不要3个一起去重
    poi_cat = poi_cat.drop_duplicates()  # 去重
    poi_cat = poi_cat.sort_values(by='venueid')
    # poi_cat.to_csv("t_split.txt", sep='\t', index=False)
    cat_candidate = torch.Tensor(poi_cat['catid'].values)  # 创建类别候选的张量

    with open('../data/cat_candidate.pk', 'wb') as f:  # 将类别候选数据序列化到文件
        pickle.dump(cat_candidate, f)

    # 建立venueID和经纬度的映射数据保存到venueid2coor_1type.pkl中
    venueid2coor = OrderedDict()
    catid2seqnum = OrderedDict()
    total_rows = len(data)
    with tqdm(total=total_rows, desc="Processing rows") as pbar:
        seq1 = 0
        seq2 = 0
        for i in range(total_rows):
            venueid = data['venueid'].values[i]
            if venueid not in venueid2coor:
                venueid2coor[venueid] = seq1
                seq1 += 1
            catid = data['catid'].values[i]
            if catid not in catid2seqnum:
                catid2seqnum[catid] = seq2
                seq2 += 1
            pbar.update(1)
    with open('../data/venueid2coor2seqnumber_1type.pkl', 'wb') as f:
        pickle.dump(venueid2coor, f)
    print("venueid2coor2seqnumber_1type.pkl Done")
    with open('../data/catid2seqnum.pkl', 'wb') as f:
        pickle.dump(catid2seqnum, f)
    print("catid2seqnum.pkl Done")

    # 3、split data into train set and test set.
    #    extract features of each session for classification
# NOTE3.数据分割为训练集和测试集，提取每个会话的特征用于分类
    vocab_size_poi = int(max(data['venueid'].values))  # 获取地点的词汇大小
    vocab_size_cat = int(max(data['catid'].values))  # 获取类别的词汇大小
    vocab_size_user = int(max(data['userid'].values))  # 获取用户的词汇大小
    vocab_size_week_n = int(max(data['week_n_list'].values))  # 获取周的词汇大小
    vocab_size_hour = int(max(data['hour'].values))

    print('vocab_size_poi: ', vocab_size_poi)
    print('vocab_size_cat: ', vocab_size_cat)
    print('vocab_size_user: ', vocab_size_user)
    print('vocab_size_week_n: ', vocab_size_week_n)
    print('vocab_size_hour:', vocab_size_hour)

# train 训练用的
    train_x = []
    train_x_cat = []
    train_x_week_n = []
    train_y = []
    train_hour = []
    train_userid = []
    train_indexs = []

# the hour and week to be predicted
    train_hour_pre = []  # 预测的小时
    train_week_pre = []  # 预测的周

# test 测试用的
    test_x = []
    test_x_cat = []
    test_x_week_n = []
    test_y = []
    test_hour = []
    test_userid = []
    test_indexs = []

# the hour and week to be predicted
    test_hour_pre = []  # 预测的小时
    test_week_pre = []  # 预测的周

    long_term = {}  # 初始化一个字典，用来存储长期特征

    long_term_feature = []  # 初始化一个列表，用于存储长期特征数据

    data_train = {}  # 初始化一个字典，用于存储训练数据
    train_idx = {}  # 初始化一个字典，用于存储所有用户的训练数据索引

    data_test = {}  # 初始化一个字典，用于存储测试数据
    test_idx = {}  # 初始化一个字典，用于存储所有用户的测试数据索引

    # 存储数据的基本信息，包括POI、类别和用户的大小
    data_train['datainfo'] = {'size_poi': vocab_size_poi+1, 'size_cat': vocab_size_cat +
                              1, 'size_user': vocab_size_user+1, 'size_week_n': vocab_size_week_n+1}  # TODO +1?

    len_session = 10  # NOTE 设置会话窗口的长度为10
    user_lastid = {}  # 初始化一个字典，用来存储用户的最后一个ID
###################################################################################
    # split data 分割数据
    # TODO 之前已经排好序了，这里还要再排吗，group要有，但这样会不会打乱时间顺序
    for uid, group in data.groupby('userid'):  # 对数据按用户ID进行分组
        data_train[uid] = {}  # 初始化该用户的训练数据集
        data_test[uid] = {}  # 初始化该用户的测试数据集
        user_lastid[uid] = []  # 初始化该用户的最后一个ID列表
        inds_u = group.index.values  # 获取该用户的所有索引值
    # NOTE 按照用户的checkins来进行分割的
        # NOTE 用80%的数据作为训练集 单用户  后期为提高精度可靠近与城市间隔测试一下
        split_ind = int(np.floor(0.8 * len(inds_u)))
        train_inds = inds_u[:split_ind]  # 训练集的索引
        test_inds = inds_u[split_ind:]  # 测试集的索引

    # get the features of POIs for user uid
        # long_term_feature.append(get_features(group.loc[train_inds]))
        # 获取用户uid的POI特征
        long_term[uid] = {}
        lt_data = group.loc[train_inds]  # 获取训练集数据
        # NOTE:长期特征：loc\hour\week\category
        long_term[uid]['loc'] = torch.LongTensor(
            lt_data['venueid'].values).cuda()  # 存储位置信息
        long_term[uid]['hour'] = torch.LongTensor(
            lt_data['hour'].values).cuda()  # 存储小时信息
        long_term[uid]['week'] = torch.LongTensor(
            lt_data['week'].values).cuda()  # 存储星期信息
        long_term[uid]['category'] = torch.LongTensor(
            lt_data['catid'].values).cuda()  # 存储类别信息

    # split the long sessions to some short ones with len_session =
    # 将长会话分割成多个短会话，每个会话长度为
        train_inds_split = []
        num_session_train = int(len(train_inds) // len_session)  # 计算训练集中会话的数量
        for i in range(num_session_train):
            train_inds_split.append(
                train_inds[i * len_session:(i + 1) * len_session])  # 分割会话
        if num_session_train < len(train_inds) / len_session:
            train_inds_split.append(train_inds[-len_session:])  # 添加剩余的训练数据

        train_id = list(range(len(train_inds_split))) 	# 生成训练集ID列表

        test_inds_split = []
        num_session_test = int(len(test_inds) // len_session)  # 计算测试集中会话的数量
        for i in range(num_session_test):
            test_inds_split.append(
                test_inds[i * len_session:(i + 1) * len_session])  # 分割会话
        if num_session_test < len(test_inds) / len_session:
            test_inds_split.append(test_inds[-len_session:])  # 添加剩余的测试数据

        # 生成测试集ID列表
        test_id = list(range(len(test_inds_split) +
                       len(train_inds_split)))[-len(test_inds_split):]
        # 测试集的索引从全局索引 [3, 4] 开始，确保了训练集和测试集的索引范围不重叠

        #
        train_idx[uid] = train_id[1:]  # 存储训练集索引（排除第一个，因为它没有历史数据）-用户的第一个
        test_idx[uid] = test_id  # 存储测试集索引

        # NOTE:generate data for comparative methods such as deepmove	为 deepmove 等比较方法生成数据，替换baselines?这部分动？
        for ind in train_id:
            if ind == 0:
                continue  # 跳过第一个索引，因为没有历史数据
            data_train[uid][ind] = {}  # 初始化该索引的训练数据
            # TODO 这个[uid][ind]是什么，ind是什么？
            history_ind = []
            for i in range(ind):
                history_ind.extend(train_inds_split[i])  # 获取当前索引之前所有会话的数据
            whole_ind = []
            whole_ind.extend(history_ind)
            whole_ind.extend(train_inds_split[ind])  # 包括当前会话的数据
            whole_data = group.loc[whole_ind]  # 获取整个数据

            loc = whole_data['venueid'].values[:-1]  # 位置数据
            tim = whole_data['hour'].values[:-1]  # 时间数据
            wek_n = whole_data['week_n_list'].values[:-1]  # 年周次数据
            target = group.loc[train_inds_split[ind]
                               [1:]]['venueid'].values  # !目标位置数据 target只是目标venueid?

            data_train[uid][ind]['loc'] = torch.LongTensor(
                loc).unsqueeze(1)  # 转换为张量并增加维度
            data_train[uid][ind]['tim'] = torch.LongTensor(
                tim).unsqueeze(1)  # 转换为张量并增加维度
            data_train[uid][ind]['wek_n'] = torch.LongTensor(
                wek_n).unsqueeze(1)
            data_train[uid][ind]['target'] = torch.LongTensor(target)  # 转换为张量
            # 目标数据traget用于计算模型损失，损失函数通常直接接受目标张量的原始形状

            user_lastid[uid].append(loc[-1])  # 更新用户的最后一个位置ID

        #! here
            group_i = group.loc[train_inds_split[ind]]  # 当前会话的数据

            # generate data for SHAN
            current_loc = group_i['venueid'].values  # 当前位置数据
            data_train[uid][ind]['current_loc'] = torch.LongTensor(
                current_loc).unsqueeze(1)  # ? 转换为张量并增加维度 -作为序列输入到模型中

            current_cat = group_i['catid'].values  # 当前类别数据
            # ?data_train[uid][ind]['current_cat'] = torch.LongTensor(current_cat).unsqueeze(1)

            current_wek_n = group_i['week_n_list'].values
            data_train[uid][ind]['current_wek_n'] = torch.LongTensor(
                current_wek_n).unsqueeze(1)  # !Test

            # generate data for my methods. X,Y,time,userid
            train_x.append(current_loc[:-1])  # 训练位置数据
            train_x_cat.append(current_cat[:-1])  # 训练类别数据
            train_x_week_n.append(current_wek_n[:-1])
            train_y.append(current_loc[1:])  # 训练目标位置数据
            train_hour.append(group_i['hour'].values[:-1])  # 训练时间数据
            train_userid.append(uid)  # 训练用户ID
            train_hour_pre.append(group_i['hour'].values[1:])  # 预测的小时数据
            train_week_pre.append(group_i['week'].values[1:])  # 预测的星期数据
            train_indexs.append(group_i.index.values)  # 训练数据索引

        ##########################
        # data for test
        for ind in test_id:
            data_test[uid][ind] = {}  # 初始化该索引的测试数据
            history_ind = []
            for i in range(len(train_inds_split)):
                history_ind.extend(train_inds_split[i])  # 获取训练集所有会话的数据
            whole_ind = []
            whole_ind.extend(history_ind)
            # 包括测试集当前会话的数据
            whole_ind.extend(test_inds_split[ind-len(train_inds_split)])

            whole_data = group.loc[whole_ind]  # 获取整个数据

            loc = whole_data['venueid'].values[:-1]  # 位置数据
            tim = whole_data['hour'].values[:-1]  # 时间数据
            wek_n = whole_data['week_n_list'].values[:-1]  # 年周次数据
            target = group.loc[test_inds_split[ind -
                                               len(train_inds_split)][1:]]['venueid'].values  # 目标位置数据

            data_test[uid][ind]['loc'] = torch.LongTensor(
                loc).unsqueeze(1)  # 将位置数据转换为张量，并增加一个维度，用于存储在测试数据字典中
            data_test[uid][ind]['tim'] = torch.LongTensor(
                tim).unsqueeze(1)  # 将时间数据转换为张量，并增加一个维度，用于存储在测试数据字典中
            data_test[uid][ind]['target'] = torch.LongTensor(
                target)  # 将目标数据转换为张量，存储在测试数据字典中
            data_test[uid][ind]['wek_n'] = torch.LongTensor(
                wek_n).unsqueeze(1)
            user_lastid[uid].append(loc[-1])  # 更新用户的最后一个访问位置

            # group_i = whole_data

            group_i = group.loc[test_inds_split[ind -
                                                len(train_inds_split)]]	 # 获取当前测试会话对应的数据

            current_loc = group_i['venueid'].values  # 获取当前会话的位置数据
            # 将当前位置数据转换为张量，并增加一个维度，存储在测试数据字典中
            data_test[uid][ind]['current_loc'] = torch.LongTensor(
                current_loc).unsqueeze(1)
            # ?上面这行为什么？-作为序列输入到模型中

            current_cat = group_i['catid'].values  # 获取当前会话的类别数据

            current_wek_n = group_i['week_n_list'].values
            data_test[uid][ind]['current_wek_n'] = torch.LongTensor(
                current_wek_n).unsqueeze(1)  # !Test

            test_x.append(current_loc[:-1])  # 将位置数据（除最后一个以外）添加到测试位置列表
            test_x_cat.append(current_cat[:-1])  # 将类别数据（除最后一个以外）添加到测试类别列表
            test_x_week_n.append(current_wek_n[:-1])
            test_y.append(current_loc[1:])  # 将目标位置数据（即下一个位置）添加到测试目标列表
            test_hour.append(group_i['hour'].values[:-1])
            test_userid.append(uid)  # 添加用户ID到测试用户ID列表
            test_hour_pre.append(group_i['hour'].values[1:])
            test_week_pre.append(group_i['week'].values[1:])
            test_indexs.append(group_i.index.values)  # 将当前测试会话的索引添加到测试索引列表

            # current_scc = group_i['sccode'].values
            # ! 这儿少了[:-1] 导致后期shape第二维变成10
            # test_x_sccode.append(current_scc[:-1])

    with open('../data/data_train.pk', 'wb') as f:  # 序列化训练数据并存储到文件
        pickle.dump(data_train, f)
    with open('../data/data_test.pk', 'wb') as f:  # 序列化测试数据并存储到文件
        pickle.dump(data_test, f)
    with open('../data/train_idx.pk', 'wb') as f:  # 序列化训练数据索引并存储到文件
        pickle.dump(train_idx, f)
    with open('../data/test_idx.pk', 'wb') as f:   # 序列化测试数据索引并存储到文件
        pickle.dump(test_idx, f)

    print('user_num: ', len(data_train.keys()))  # 打印训练数据中用户的数量
    # minMax = MinMaxScaler()
    # long_term_feature = minMax.fit_transform(np.array(long_term_feature))

    with open('../data/long_term.pk', 'wb') as f:  # 序列化长期特征数据并存储到文件
        pickle.dump(long_term, f)

    # with open('long_term_feature.pk','wb') as f:
    # pickle.dump(long_term_feature,f)

    # 检查函数：用于确保所有输入列表的长度一致
    def check_lengths(*arrays):
        reference_length = len(arrays[0])
        for index, array in enumerate(arrays, start=1):
            if len(array) != reference_length:
                print(
                    f"输入数据长度不一致：第 {index} 个输入的长度为 {len(array)}，预期长度为 {reference_length}")
                return False
        return True

    # 定义一个函数，用于创建数据加载器
    def dataloader(X, X_cat, X_wekn, Y, hour, userid, hour_pre, week_pre):
        # 检查输入数据的一致性
        if not check_lengths(X, X_cat, X_wekn, Y, hour, userid, hour_pre, week_pre):
            raise ValueError("输入数据长度不一致，无法继续进行。")
        try:
            # 转换为 NumPy 数组
            X = np.array(X)
            X_cat = np.array(X_cat)
            X_wekn = np.array(X_wekn)
            Y = np.array(Y)
            hour = np.array(hour)
            userid = np.array(userid)
            hour_pre = np.array(hour_pre)
            week_pre = np.array(week_pre)
        except ValueError as e:
            print("数据转换为 NumPy 数组时出错：", e)
            # 打印具体错误的列表长度以进行调试
            print("X  :  ", [len(x) for x in X])
            print("X_cat:", [len(x) for x in X_cat])
            print("X_wekn:", [len(x) for x in X_wekn])
            print("Y:", len(Y))
            print("Y:", [len(y) for y in Y])
            print("hour:", len(hour))
            print("userid:", len(userid))
            print("hour_pre:", len(hour_pre))
            print("week_pre:", len(week_pre))
            raise

        torch_dataset = Data.TensorDataset(torch.LongTensor(X), torch.LongTensor(X_cat), torch.LongTensor(X_wekn), torch.LongTensor(
            Y), torch.LongTensor(hour), torch.LongTensor(userid), torch.LongTensor(hour_pre), torch.LongTensor(week_pre))

        # 分成小批次（batches）
        loader = Data.DataLoader(
            dataset=torch_dataset,  # 设置数据集
            batch_size=batch_size,  # 设置批处理大小
            shuffle=True,  # 设置为随机打乱数据			#TODOshuffle #BUG是否随机打乱,测试对结果的影响
            # shuffle=True 表示在每个epoch开始时，整个数据集的样本顺序将被随机打乱
            num_workers=0,  # 设置工作线程数
        )
        return loader  # 返回数据加载器

    # 创建训练数据加载器
    loader_train = dataloader(train_x, train_x_cat, train_x_week_n,
                              train_y, train_hour, train_userid, train_hour_pre, train_week_pre)

    loader_test = dataloader(test_x, test_x_cat, test_x_week_n,
                             test_y, test_hour, test_userid, test_hour_pre, test_week_pre)

    pre_data = {}  # 初始化一个字典，用于存储预处理数据
    pre_data['size'] = [vocab_size_poi, vocab_size_cat, vocab_size_week_n,
                        vocab_size_user, len(train_x), len(test_x)]  # 存储相关尺寸信息
    pre_data['loader_train'] = loader_train  # 存储训练数据加载器
    pre_data['loader_test'] = loader_test  # 存储测试数据加载器

    with open('../data/pre_data.txt', 'wb') as f:  # 序列化预处理数据并存储到文件
        pickle.dump(pre_data, f)
    return pre_data  # 返回预处理数据


def geodistance(lng1, lat1, lng2, lat2):
    lng1, lat1, lng2, lat2 = map(
        radians, [float(lng1), float(lat1), float(lng2), float(lat2)])
    dlon = lng2-lng1
    dlat = lat2-lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance = 2*asin(sqrt(a))*6371*1000
    distance = round(distance/1000, 3)
    return distance


def caculate_poi_distance(poi_coors, start_x_index=0, start_y_index=0):
    print("distance matrix")
    print(len(poi_coors))

    epsilon = 1e6  # 一个很小的正数，避免除零错误
    sim_matrix = np.zeros((len(poi_coors) + 1, len(poi_coors) + 1))
    try:
        #with open('../data/datadistance_1type.txt', 'w') as f:
        max_dis = 0
        for i in tqdm(range(len(poi_coors))):
            for j in tqdm(range(i, len(poi_coors)), leave=False):
                poi_current = i
                poi_target = j
                poi_current_coor = poi_coors[poi_current]
                poi_target_coor = poi_coors[poi_target]
                if poi_current == poi_target:
                    distance_between = epsilon  # 设置对角线上的距离为一个很大的正数
                else:
                    distance_between = geodistance(
                        poi_current_coor[0], poi_current_coor[1], poi_target_coor[0], poi_target_coor[1])
                    if distance_between > max_dis:  max_dis = distance_between
                sim_matrix[poi_current][poi_target] = distance_between
                sim_matrix[poi_target][poi_current] = distance_between
                #f.write(f"{poi_current},{poi_target},{distance_between}\n")
        print("max_distance_between_poi: ",max_dis)
    except Exception as e:
        print(f"Error occurred: {e}")
    # pickle.dump(sim_matrix, open('../data/datadistance_1type.pkl', 'wb'))
    print("already save the distance matrix")

    return sim_matrix


def excute_caculate_poi_distance(df):
    print("function: excute_caculate_poi_distance is running!")
    # df = pd.read_table('../data/user_1type_all.txt', header=None)
    df = df.drop_duplicates(subset=[1], keep='first')
    poi_coordinate = [(float(row[3]), float(row[4]))
                      for _, row in df.iterrows()]
    # unique_poi_coordinate = list(set(poi_coordinate))
    coorlen = len(poi_coordinate)
    print("coorlen: ", coorlen)
    start_x_index = 0
    start_y_index = 0
    sim_matrix = caculate_poi_distance(
        poi_coordinate, start_x_index, start_y_index)
    print("poi_distance_matrix.shape: ",sim_matrix.shape)
    pickle.dump(sim_matrix, open('../data/datadistance_1type_2.pkl', 'wb'))

    '''
    # 中断恢复
    # line_count = sum(1 for line in open('../data/datadistance_1type.txt', 'r'))
    # s_x = line_count % coorlen
    # s_y = line_count - s_x*coorlen - 1
    # sim_matrix = caculate_poi_distance(
    #     unique_poi_coordinate, s_x+1, s_y)
    # pickle.dump(sim_matrix, open('../data/datadistance_1type.pkl', 'wb'))

    # 读取txt文件中的计算结果保存到pkl文件中
    '''


def caculate_venue_frequency(df_vid):
    # Step 1: 读取和预处理数据
    print(df_vid.head())
    df_vid_unique = df_vid.drop_duplicates(subset=[1], keep='first')
    # TODO .loc[:,]到底什么用，catid_time_matrix那边不行🙅啊
    df_vid.loc[:, 5] = pd.to_datetime(df_vid.loc[:, 5])
    print("df_vid len: ", len(df_vid))
    # 2012和2013都是都52周 1-52 2 0-51
    df_vid.loc[:, 7] = df_vid.loc[:, 5].dt.isocalendar().week

    # Step 2: 计算每个 VenueID 在每周的访问频次
    venue_freq_matrix = np.zeros((53, len(df_vid_unique) + 1))
    venue_dict = {}
    total_rows = len(df_vid)
    with tqdm(total=total_rows, desc="Processing rows") as pbar:
        i = 0
        for _, row in df_vid.iterrows():
            week = row[7]
            if row[1] not in venue_dict:
                venue_dict[row[1]] = i
                venue_freq_matrix[week][i] += 1
                i += 1
            else:
                venue_freq_matrix[week][venue_dict[row[1]]] += 1
            pbar.update(1)
    print("venue_dict len: ", len(venue_dict))
    print("venue_freq_matrix shape: ", venue_freq_matrix.shape)

    return venue_freq_matrix


def excute_caculate_venue_frequency(df):
    # df_vid = pd.read_table('../data/user_1type_all.txt', header=None)
    df_vid = df
    venue_freq_matrix = caculate_venue_frequency(df_vid)
    print("venue_freq_matrix.shape: ",venue_freq_matrix.shape)
    pickle.dump(venue_freq_matrix, open('../data/venue_freq_matrix.pkl', 'wb'))
    print("venue_freq_matrix.pkl is Done")
    # 将 np 矩阵 venue_freq_matrix 保存到 txt 文件
    np.savetxt('../data/venue_freq_matrix.txt',
               venue_freq_matrix, fmt='%d', delimiter='\t')
    print("venue_freq_matrix.txt is Done")


def caculate_time_cid(df):
    df_catid_unique = df.drop_duplicates(subset=[2], keep='first')
    print("len df_catid_unique: ", len(df_catid_unique))
    sim_matrix = np.zeros((48, len(df_catid_unique)+1))  # TODO 48,242

    df[5] = pd.to_datetime(df[5], errors='coerce')
    # # 检查转换是否成功
    # if df.loc[:, 5].isnull().any():
    #     raise ValueError("Error in converting column 5 to datetime format")
    df[7] = df[5].dt.hour
    df[8] = df[5].dt.weekday

    catid_dict = {}
    total_rows = len(df)
    with tqdm(total=total_rows, desc="Processing rows") as pbar:
        i = 0
        for _, row in df.iterrows():
            hour = row[7]  # 0-47
            if row[8] == 0 or row[8] == 6:
                hour += 24
            if row[2] not in catid_dict:
                catid_dict[row[2]] = i
                sim_matrix[hour][i] += 1
                i += 1
            else:
                sim_matrix[hour][catid_dict[row[2]]] += 1
            pbar.update(1)
    print(sim_matrix.shape)
    return sim_matrix


def excute_caculate_time_cid(df):
    # df = pd.read_table('../data/user_1type_all.txt', header=None)
    catid_time_matrix = caculate_time_cid(df)
    pickle.dump(catid_time_matrix, open('../data/catid_time_matrix.pkl', 'wb'))
    print("catid_time_matrix.pkl is Done")
    # 将 np 矩阵 catid_time_matrix 保存到 txt 文件
    np.savetxt('../data/catid_time_matrix.txt',
               catid_time_matrix, fmt='%d', delimiter='\t')
    print("catid_time_matrix.txt is Done")
