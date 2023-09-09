# 使用fasttext-0.9.2-cp310-cp310-win_amd64.whl安装fasttext
import fasttext

"""
超参数设定：
无监督训练模式"skipgram"或"cbow"，默认是"skipgram"，该模式实践中利用子词方面比"cbow"好
词嵌入维度dim默认100
训练epoch默认5
学习率lr默认0.05
训练使用cpu线程数默认12
"""
model = fasttext.train_unsupervised('../enwik9/output.txt')

print(type(model))

# 显示训练后的词向量
print(model.get_word_vector("the"))
print(model.get_word_vector("the").shape)

# 显示词向量的邻近词向量
print(model.get_nearest_neighbors('we'))

# 保存模型至项目的根目录
model.save_model('trained_skipgram_model.bin')