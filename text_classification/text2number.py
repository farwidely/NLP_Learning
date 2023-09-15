import numpy as np
import torch
from transformers import AutoTokenizer
from datasets import load_dataset

train_dataset = load_dataset("ag_news", split="train")
test_dataset = load_dataset("ag_news", split="test")

print(train_dataset)
print(test_dataset)

print(train_dataset[0])

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 把文本转换为数值
transformed_train_dataset = tokenizer(train_dataset['text'], padding=True, truncation=True, return_tensors="pt")
transformed_test_dataset = tokenizer(test_dataset['text'], padding=True, truncation=True, return_tensors="pt")
transformed_train_dataset['label'] = torch.tensor(train_dataset['label'])
transformed_test_dataset['label'] = torch.tensor(test_dataset['label'])

print(transformed_train_dataset)
print(type(transformed_train_dataset))
print(transformed_train_dataset['input_ids'].shape)
print(transformed_train_dataset['label'].shape)
print(transformed_test_dataset['input_ids'].shape)
print(transformed_test_dataset['label'].shape)

# 创建用于存储数据集的字典
dataset = {'train_text': np.array(transformed_train_dataset['input_ids']),
           'train_label': np.array(transformed_train_dataset['label']),
           'test_text': np.concatenate((np.array(transformed_test_dataset['input_ids']),
                                        np.zeros((7600, 102))), axis=1),
           'test_label': np.array(transformed_test_dataset['label'])}

np.savez('agnews_number_dataset', **dataset)
