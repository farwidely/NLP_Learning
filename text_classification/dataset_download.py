# 使用huggingface下载数据集
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader

# 下载数据集
dataset = load_dataset('ag_news')
print(dataset)

# 获取训练集和测试集
train_dataset = dataset['train']
test_dataset = dataset['test']

# 查看数据集长度
train_data_size = len(train_dataset)
test_data_size = len(test_dataset)
print(f"训练数据集的长度为: {train_data_size}")
print(f"测试数据集的长度为: {test_data_size}")

# 定义批处理大小和工作进程数
batch_size = 1

# 创建训练集和测试集的DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# # 显示train_dataloader中的第一份数据
# print(next(iter(train_dataloader)))

train_sentence_length = []
test_sentence_length = []

for data in train_dataloader:
    # print(type(data))
    # print(data)

    text = data['text']
    # print(text)
    # print(len(text[0].split()))
    train_sentence_length.append(len(text[0].split()))

    # label = data['label']
    # print(label)
    # print(len(label))


for data in test_dataloader:
    text = data['text']
    test_sentence_length.append(len(text[0].split()))

np.save('train_sentence_length', train_sentence_length)
np.save('test_sentence_length', test_sentence_length)
