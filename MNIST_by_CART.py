# -*- coding: utf-8 -*-
# 使用CART进行MNIST手写数字分类
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# 加载手写数字
digits = load_digits()
data = digits.data

# 打印出手写数字数据集的规模和维度
print(data.shape)

# 查看第一幅图像对应的二维数组
print(digits.images[0])

# 查看第一幅图片对应的标记
print(digits.target[0])

# 将图像转化成灰度图、显示图片格式、展示图片
plt.gray()
plt.imshow(digits.images[0])
plt.show()

# 将数据集中的25%作为测试集
train_x, test_x, train_y, test_y = train_test_split(data, digits.target, test_size=0.25, random_state = 33)

# 进行数据归一化和标准化,并将数据集转换成标准正态分布的数据
ss = preprocessing.StandardScaler()
train_ss_x = ss.fit_transform(train_x)
test_ss_x = ss.transform(test_x)

# CART分类
CART_model = DecisionTreeClassifier()
CART_model.fit(train_ss_x, train_y)
predict_y = CART_model.predict(test_ss_x)
print('CART准确率: %0.4lf' % accuracy_score(predict_y, test_y))