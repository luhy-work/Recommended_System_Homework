from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB
from tpot import TPOTClassifier
import numpy as np
import pandas as pd

# 数据加载
train_data = pd.read_csv(r"D:\pythonData\L2_data\titan\train.csv")
test_data = pd.read_csv(r"D:\pythonData\L2_data\titan\test.csv")

# 数据探索
# 查看train_data信息
pd.set_option('display.max_columns', None)

print('查看数据名称: 列名, 非空个数, 类型等等')
print(train_data.info())
print("-" * 30)

print("数据摘要")
print(train_data.describe())
print("-" * 30)

print('查看离散数据分布')
print(train_data.describe(include=[np.object]))
print('-' * 30)

print('查看前5条数据')
print(train_data.head())
print('-' * 30)

print('查看后5条数据')
print(train_data.tail())


# 使用平均年龄来填充年龄中的NaN值
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)

# 使用票价的均价来填充票价中的NaN值
train_data['Fare'].fillna(train_data['Fare'].mean(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)

# 使用登陆最多的港口来填充登陆港口中的NaN值
print(train_data['Embarked'].value_counts())
train_data['Embarked'].fillna('S', inplace=True)
test_data['Embarked'].fillna('S', inplace=True)

print(train_data.info())

# 特征选择
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
train_features = train_data[features]
train_labels = train_data['Survived']
test_features = test_data[features]
print('特征值')
print(train_features)

dvec = DictVectorizer(sparse=False)
train_features = dvec.fit_transform(train_features.to_dict(orient='record'))
print(dvec.feature_names_)

# 构造ID3决策树
clf = DecisionTreeClassifier(criterion='entropy')
# 决策树训练
clf.fit(train_features, train_labels)

# 决策树预测
test_features = dvec.transform(test_features.to_dict(orient='record'))
pred_labels = clf.predict(test_features)

# 决策树的准确率(基于训练集)
acc_decision_tree = round(clf.score(train_features, train_labels), 6)
print(u'score准确率为 %.4lf' % acc_decision_tree)

# 使用k折交叉验证 统计决策树准确率
print(u'cross_val_score准确率为 %.4lf' % np.mean(cross_val_score(clf, train_features, train_labels, cv=10)))


# knn分类器
knn_model = KNeighborsClassifier()
knn_model.fit(train_features, train_labels)
knn_pred_labels = knn_model.predict(test_features)
knn_result = pd.concat([test_data, pd.DataFrame(knn_pred_labels)], axis=1)
knn_result.rename({0: 'KNN_Survived'}, axis=1, inplace=True)
print(knn_result)
#
# print('-' * 40)

# Bayes分类器
bayes_model = BernoulliNB()
bayes_model.fit(train_features, train_labels)
bayes_pred_labels = bayes_model.predict(test_features)
bayes_result = pd.concat([test_data, pd.DataFrame(bayes_pred_labels)], axis=1)
bayes_result.rename({0: 'bayes_Survived'}, axis=1, inplace=True)
print(bayes_result)

# TPOT自动机器学习
tpot_train_x = train_data[features]
tpot_train_y = train_data['Survived']
dvec = DictVectorizer(sparse=False)
new_train_x = dvec.fit_transform(tpot_train_x.to_dict(orient='record'))
x_train, x_test, y_train, y_test = train_test_split(new_train_x.astype(np.float64),
                                                    tpot_train_y.astype(np.float64), test_size=0.25)
tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2)
tpot.fit(x_train, y_train)
print(tpot.score(x_test, y_test))
tpot.export('titanic_TPOT_pipeline.py')
