# 添加n-gram特征是常见的文本特征处理方法
# n个词共同出现且相邻的情况成为n-gram特征，常用n=2，n=3


ngram_rang = 2


def create_ngram_set(input_list):
    return set(zip(*[input_list[i:] for i in range(ngram_rang)]))


input = [1, 3, 2, 1, 5, 3]
result = create_ngram_set(input)
print(result)