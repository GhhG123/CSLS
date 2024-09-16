import pandas as pd
# from train_long_short import batch_size
import utils
# TODOfrom train import batch_size
import json

# 读取 JSON 文件
with open('paras.json', 'r') as f:
    config = json.load(f)

file_path_name = config['dataset_path']
batch_size = config['batch_size']
print("file_path_name: ",file_path_name)
print("batch_size: ",batch_size)

print("start preprocess")   # 开始预处理
# 调用预处理函数---**********@@@@@@@@@+++++++++++++++++========

# df = pd.DataFrame(pd.read_table(file_path_name, header=None))
# utils.excute_caculate_poi_distance(df)
# utils.excute_caculate_time_cid(df)
# utils.excute_caculate_venue_frequency(df)


# 读取并创建数据框架
data = pd.DataFrame(pd.read_table(file_path_name, header=None, encoding="latin-1"))
# 设置数据列名
data.columns = ["userid", "venueid", "catid", "latitude",
                "longitude", "timestamp", "sccode"]   #tb_add: #, "sccode"
pre_data = utils.sliding_varlen(data, batch_size)

print("pre done")
