import jieba
import jieba.posseg as pseg


content1 = "故事的小黄花从出生那年就飘着童年的荡秋千随记忆一直晃到现在"

# 精确模式分词（默认），每个字不会重复，适合文本分析
# 返回一个生成器对象
result1 = jieba.cut(content1, cut_all=False)  # cut_all默认为False
print(result1)

# 返回分词列表
result2 = jieba.lcut(content1, cut_all=False)
print(result2)


# 全模式分词模式，把所有可能的组合扫描出来，速度非常快，但是不能消除歧义
# 返回一个生成器对象
result3 = jieba.cut(content1, cut_all=True)
print(result3)

# 返回分词列表
result4 = jieba.lcut(content1, cut_all=True)
print(result4)


# 搜索引擎模式分词，在精确模式的基础上，对长词再次划分，提高召回率，适用于搜索引擎分词
# 返回一个生成器对象
result5 = jieba.cut_for_search(content1)
print(result5)

# 返回分词列表
result6 = jieba.lcut_for_search(content1)
print(result6)


content2 = "让那彩虹长桥无限伸展飞象日日云上表演魔幻现实寻到相交点"
print(jieba.lcut(content2))
# 自定义词典书写格式：”关键词 词频 词性“，词频和词性可省略
jieba.load_userdict("./user_dict.txt")
print(jieba.lcut(content2))


# 中文词性标注
print(pseg.lcut("孤单听雨的猫往时间裂缝里看到了我"))
