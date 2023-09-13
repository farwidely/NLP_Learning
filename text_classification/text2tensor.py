import numpy as np
import torch
from transformers import AutoTokenizer
from datasets import load_dataset

train_dataset = load_dataset("ag_news", split="train")
test_dataset = load_dataset("ag_news", split="test")

print(train_dataset)
print(test_dataset)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def convert_to_tensors(example):
    inputs = tokenizer(example["text"], padding=True, truncation=True, return_tensors="pt")
    inputs["label"] = torch.tensor(example["label"])
    return inputs


# 把文本转换为数值
transformed_train_dataset = train_dataset.map(convert_to_tensors)
transformed_test_dataset = test_dataset.map(convert_to_tensors)

dataset = {'train': {'text': np.empty(0), 'label': np.empty(0)}, 'test': {'text': np.empty(0), 'label': np.empty(0)}}

transformed_text = np.array(transformed_test_dataset[0]["input_ids"])
transformed_text = transformed_text[0][1:-1]
print(transformed_text)
print(type(transformed_text))
print(transformed_text.shape)
dataset['test']['text'].append(transformed_text)
