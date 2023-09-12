# 实现新闻分类任务
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from text_classification.model import TextSentiment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64

# 下载数据集
dataset = load_dataset('ag_news')

# 获取训练集和测试集
train_dataset = dataset['train']
test_dataset = dataset['test']


# 创建训练集和测试集的DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

VOCAB_SIZE = len(train_dataset.get_vocab())
RMBED_DIM = 32
NUM_CLASS = len(train_dataset.get_label())
model = TextSentiment(VOCAB_SIZE, RMBED_DIM, NUM_CLASS)
model.to(device)

