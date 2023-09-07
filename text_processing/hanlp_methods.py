# hanlp可进行中英文分词，但是中文分词效果可能没有jieba好
import hanlp


# 加载CTB_CONVSEG预训练模型进行分词任务
tokenizer = hanlp.load('CTB6_CONVSEG')
result1 = tokenizer("故事的小黄花从出生那年就飘着童年的荡秋千随记忆一直晃到现在")
print(result1)

# 英文分词直接使用规则方法
tokenizer2 = hanlp.utils.rules.split_sentence
result2 = tokenizer2("we were both young when I first saw you")
print(result2)


# 中文命名实体识别
# 加载中文命名实体识别预训练模型MSRA_NER_BERT_BASE_ZH
recognizer = hanlp.load("MSRA_NER_BERT_BASE_ZH")
print(recognizer(list("塞纳河畔左岸的咖啡")))


# 中文词性标注
tagger1 = hanlp.load('CTB5_POS_RNN_FASTTEXT_ZH')
print(tagger1(list("孤单听雨的猫往时间裂缝里看到了我")))

# 英文词性标注
tagger2 = hanlp.load("PTB_POS_RNN_FASTTEXT_EN")
print(tagger2("Can I go where you go".split()))
