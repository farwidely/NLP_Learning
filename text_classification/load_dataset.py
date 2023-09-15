import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

dataset = np.load('agnews_number_dataset.npz')
print(dataset['train_text'].shape)
print(dataset['train_label'].shape)
print(dataset['test_text'].shape)
print(dataset['test_label'].shape)


class AG_NEWS_Dataset(Dataset):
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
