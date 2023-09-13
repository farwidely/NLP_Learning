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

dataset = {'train': {'text': [], 'label': []}, 'test': {'text': [], 'label': []}}

list = transformed_test_dataset[0]["input_ids"]
print(list)
print(type(list))