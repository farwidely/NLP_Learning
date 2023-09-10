import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

train_data = pd.read_csv("./cn_data/train.tsv", sep="\t")
valid_data = pd.read_csv("./cn_data/dev.tsv", sep="\t")

print(train_data.head())

# 获得训练数据标签数量分布
sns.countplot(x="label", data=train_data)
plt.title("train_data")
plt.show()

# 获得验证数据标签数量分布
sns.countplot(x="label", data=valid_data)
plt.title("valid_data")
plt.show()

# 在数据中添加表示句子长度的列
train_data["sentence_length"] = list(map(lambda x: len(x), train_data["sentence"]))
valid_data["sentence_length"] = list(map(lambda x: len(x), valid_data["sentence"]))

# 绘制训练集句子长度列的数量分布图
sns.countplot(x="sentence_length", data=train_data)

# 关闭横坐标
plt.xticks([])
plt.show()

# 绘制训练集句子长度分布图
sns.distplot(train_data["sentence_length"])

# 关闭纵坐标
plt.yticks([])
plt.show()

# 绘制测试集句子长度列的数量分布图
sns.countplot(x="sentence_length", data=valid_data)

# 关闭横坐标
plt.xticks([])
plt.show()

# 绘制测试集句子长度分布图
sns.distplot(valid_data["sentence_length"])

# 关闭纵坐标
plt.yticks([])
plt.show()

# 绘制训练集长度分布散点图
sns.stripplot(y="sentence_length", x="label", data=train_data)
plt.show()

# 绘制验证集
sns.stripplot(y="sentence_length", x="label", data=valid_data)
plt.show()

