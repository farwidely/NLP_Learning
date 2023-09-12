# 规范化文本句子长度，过长的截断，过短的补0
from keras.preprocessing import sequence


# 设定长度标准
cutlen = 10


def padding(x):
    return sequence.pad_sequences(x, cutlen)


x_train = [[1, 23, 5, 32, 55, 63, 2, 21, 78, 32, 23, 1], [2, 32, 1, 23, 1]]

result = padding(x_train)
print(result)