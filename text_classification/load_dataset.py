import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class AG_NEWS_Dataset(Dataset):
    # 构建pytorch数据集
    def __init__(self, text, labels):
        super(AG_NEWS_Dataset, self).__init__()
        self.data = torch.from_numpy(text)
        self.labels = torch.from_numpy(labels)

    def __getitem__(self, index):
        text = self.data[index]
        label = self.labels[index]
        return text, label

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':

    # 导入转化为数值的文本数据集
    dataset = np.load('agnews_number_dataset.npz')
    print(dataset['train_text'].shape)
    print(dataset['train_label'].shape)
    print(dataset['test_text'].shape)
    print(dataset['test_label'].shape)
    # 获取不同的单词的数量
    print(len(set(dataset['train_text'].flatten()).union(set(dataset['test_text'].flatten()))))
    # 获取句子长度
    print(len(dataset['train_text'][0]))
    # 获取label的数量
    print(len(set(dataset['test_label'])))

    train_dataset = AG_NEWS_Dataset(dataset['train_text'], dataset['train_label'])
    test_dataset = AG_NEWS_Dataset(dataset['test_text'], dataset['test_label'])

    train_dataloader = DataLoader(train_dataset, batch_size=2)

    # 测试数据的读取
    for data in train_dataloader:
        text, labels = data
        print(text)
        print(labels)
        break
