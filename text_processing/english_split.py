# hanlp可进行中英文分词，但是中文分词效果可能没有jieba好
import hanlp


# 加载CTB_CONVSEG预训练模型进行分词任务
tokenizer = hanlp.load('CTB6_CONVSEG')
print(tokenizer("故事的小黄花从出生那年就飘着童年的荡秋千随记忆一直晃到现在"))
print(tokenizer("we were both young when I first saw you"))