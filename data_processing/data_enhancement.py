# 回译数据增强法是常用的文本数据增强方法
# 回译数据增强法是指将文本数据翻译成另一种语言（一般选择小语种），之后再翻译回原语言，即可认为得到与原文本同标签的新文本，一般基于google翻译接口
from translate import Translator


sample = "还记得你说家是唯一的城堡，随着稻香河流继续奔跑微微笑，小时候的梦我知道。"


translator1 = Translator(to_lang='fr', from_lang='zh-cn')

translations1 = translator1.translate(sample)
print("中间翻译结果：", translations1)


translator2 = Translator(to_lang='zh-cn', from_lang='fr')

translations2 = translator2.translate(translations1)
print("回译结果：", translations2)
