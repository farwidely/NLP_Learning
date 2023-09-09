from sklearn.preprocessing import OneHotEncoder


# 创建 OneHotEncoder 对象
encoder = OneHotEncoder()

# 定义类别标签列表
categories = ["周杰伦", "陈奕迅", "林俊杰", "邓紫棋", "王力宏", "蔡依林"]

# 将类别标签列表转换为二维数组
categories_2d = [[label] for label in categories]
print(categories_2d)

# 进行 One-Hot 编码
one_hot_encoded = encoder.fit_transform(categories_2d).toarray()
print(one_hot_encoded)
print(type(one_hot_encoded))
