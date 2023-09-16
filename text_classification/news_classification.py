# 实现新闻分类任务
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from text_classification.load_dataset import AG_NEWS_Dataset
from text_classification.model import TextSentiment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置batch_size
batch_size = 64

# 导入转化为数值的文本数据集
dataset = np.load('agnews_number_dataset.npz')
# 初始化训练集和测试集
train_dataset = AG_NEWS_Dataset(dataset['train_text'], dataset['train_label'])
test_dataset = AG_NEWS_Dataset(dataset['test_text'], dataset['test_label'])

# 初始化dataloader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# 获取不同的单词的数量
VOCAB_SIZE = len(set(dataset['train_text'].flatten()).union(set(dataset['test_text'].flatten())))
# 设置词嵌入维度
EMBED_DIM = 32
# 获取label的数量
NUM_CLASS = len(set(dataset['test_label']))

# 初始化模型
model = TextSentiment(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM)
model.to(device)

# 初始化损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

# 初始化优化器
learning_rate = 1e-2
momentum = 0.9
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

total_train_step = 0
total_test_step = 0
epoch = 30

start = time.time()

for i in range(epoch):
    print(f"------第 {i + 1} 轮训练开始------")

    start1 = time.time()

    # 训练步骤开始
    model.train()
    for data in tqdm(train_dataloader):
        text, targets = data
        text = text.to(device)
        targets = targets.to(device)
        outputs = model(text)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print(f"训练次数: {total_train_step}，Loss: {loss.item()}")

    end1 = time.time()
    print(f"本轮训练时长为{end1 - start1}秒")

    start2 = time.time()

    # 测试步骤开始
    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in tqdm(test_dataloader):
            text, targets = data
            text = text.to(device)
            targets = targets.to(device)

            outputs = model(text)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()

            total_accuracy += accuracy

    print(f"整体测试集上的Loss: {total_test_loss}")
    print(f"整体测试集上的正确率: {total_accuracy / len(dataset['test_label'])}")

    end2 = time.time()
    print(f"本轮测试时长为{end2 - start2}秒\n")

    total_test_step += 1

    if i == 29:
        torch.save(model, f"./trained_model_gpu_30.pth")
        print("模型已保存")

end = time.time()
print(f"训练+测试总时长为{end - start}秒")
