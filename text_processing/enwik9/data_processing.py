from bs4 import BeautifulSoup

# 读取enwik9数据集
with open('enwik9', 'r', encoding='utf-8') as file:
    data = file.read()

# 使用BeautifulSoup解析HTML/XML
soup = BeautifulSoup(data, 'html.parser')

# 提取纯文本内容
text = soup.get_text()

# # 打印纯文本内容
# print(type(text))

# 保存为txt
with open("output.txt", "w", encoding="utf-8") as file:
    file.write(text)
