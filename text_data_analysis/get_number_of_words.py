import jieba
from itertools import chain
import pandas as pd
from matplotlib import pyplot as plt

plt.style.use('fivethirtyeight')

train_data = pd.read_csv("./cn_data/train.tsv", sep="\t")
valid_data = pd.read_csv("./cn_data/dev.tsv", sep="\t")

print(train_data.head())


train_vocab = set(chain(*map(lambda x: jieba.lcut(x), train_data["sentence"])))
print("训练集中相异的词汇总数为：", len(train_vocab))

valid_vocab = set(chain(*map(lambda x: jieba.lcut(x), valid_data["sentence"])))
print("验证集中相异的词汇总数为：", len(valid_vocab))
