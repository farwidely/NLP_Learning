# 高频形容词词云
from itertools import chain
import jieba.posseg as pseg
from wordcloud import WordCloud
import pandas as pd
from matplotlib import pyplot as plt

plt.style.use('fivethirtyeight')

train_data = pd.read_csv("./cn_data/train.tsv", sep="\t")
valid_data = pd.read_csv("./cn_data/dev.tsv", sep="\t")

print(train_data.head())


def get_a_list(text):
    # 使用pseg获得词性元组，筛选形容词
    r = []
    for g in pseg.lcut(text):
        if g.flag == "a":
            r.append(g.word)
    return r


def get_word_cloud(keywords_list):
    # 绘制词云
    # font_path为字体，max_words为显示的词数量
    wordcloud = WordCloud(font_path="./msyh.ttf", max_words=100, background_color='white')
    # 将传入的列表转化为字符串格式
    keywords_string = " ".join(keywords_list)
    # 生成词云
    wordcloud.generate(keywords_string)

    # 绘制图像
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


# 获得训练集正样本
p_train_data = train_data[train_data["label"] == 1]["sentence"]

# 获得训练集正样本的形容词
train_p_a_vocab = chain(*map(lambda x: get_a_list(x), p_train_data))
print(train_p_a_vocab)

# 获得训练集负样本
n_train_data = train_data[train_data["label"] == 0]["sentence"]

# 获得训练集负样本的形容词
train_n_a_vocab = chain(*map(lambda x: get_a_list(x), n_train_data))
print(train_p_a_vocab)

# 绘制词云
get_word_cloud(train_p_a_vocab)
get_word_cloud(train_n_a_vocab)


# 获得验证集正样本
p_valid_data = valid_data[valid_data["label"] == 1]["sentence"]

# 获得验证集正样本的形容词
valid_p_a_vocab = chain(*map(lambda x: get_a_list(x), p_valid_data))
print(valid_p_a_vocab)

# 获得验证集负样本
n_valid_data = valid_data[valid_data["label"] == 0]["sentence"]

# 获得验证集负样本的形容词
valid_n_a_vocab = chain(*map(lambda x: get_a_list(x), n_valid_data))
print(valid_p_a_vocab)


# 绘制词云
get_word_cloud(valid_p_a_vocab)
get_word_cloud(valid_n_a_vocab)
