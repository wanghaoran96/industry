import os
import json
import moxing as mox
import torch
from naie.context import Context
from naie.datasets import get_data_reference

from torch import nn

# from utils.model_utils import read_data

get_data_reference(dataset="Mnist", dataset_entity="Mnist", enable_local_cache=True)  # load data


def read_data(dataset, subset='data'):
    '''parses data in given train and test data directories
    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''

    # 路径为 数据集名称、数据实体名称、data文件夹省略，接文件路径
    train_data_dir = os.path.join('/cache/datasets/Mnist/Mnist', dataset, subset, 'train')
    test_data_dir = os.path.join('/cache/datasets/Mnist/Mnist', dataset, subset, 'test')
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    clients = list(sorted(train_data.keys()))
    print('train data dist:\n', [len(x['y']) for x in train_data.values()])
    return clients, groups, train_data, test_data


# for root, dirs, files in os.walk('/cache/datasets'):
#     print("----root-----")
#     print(root)
#
#     print("---dir---")
#     for name in dirs:
#         file_name = os.path.join(root, name)
#         print(file_name)
#
#     print("---file---")
#     for name in files:
#         file_name = os.path.join(root, name)
#         print(file_name)

'''
1、保存模型参数到内部状态字典
'''
# data = read_data("Mnist")
# print("users:", len(data[0]))

# params
p = Context.get("param")
print("p:", p, type(p))
# 定义类
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()  # 将连续的DIM范围展平为张量
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )  # 一个有序的模块容器。数据按照定义的顺序通过所有模块。可以使用顺序容器来组合一个快速网络

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# PyTorch 模型将学习到的参数存储在内部状态字典中，称为state_dict. 这些可以通过以下torch.save 方法持久化：

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
model = NeuralNetwork().to(device)
torch.save(model.state_dict(), '/cache/model_weights.pkl')
# os.listdir("/cache")
# with open("/cache/model_weights.pth", "wb") as f:
#     f.write(b"ok")
mox.file.copy('/cache/model_weights.pkl', os.path.join(Context.get_output_path(), 'model_weights.pkl'))
