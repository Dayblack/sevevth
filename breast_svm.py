# -*- coding: utf-8 -*-
# 乳腺癌诊断分类----SVM
"""
流程简述：
1.首先我们需要加载数据源；
2.准备阶段，对加载的数据源进行探索，查看样本特征和特征值，并数据可视化。“完全合一”的
准则评估数据的质量进行数据清洗。做特征选择、特征工程，最后进行模型训练。
3.选用高斯核函数进行分类，最后评估。
"""
#根据乳腺肿块数字化图像检验结果，建立SVM分类器
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

# 加载数据集
data = pd.read_csv(r"C:\Users\ASUS\Desktop\work\data\data.csv")

# 数据探索
# 查看数据集中列，将dataframe中的列全部显示出来
pd.set_option('display.max_columns', None)
print(data.columns)
print(data.head(5))
print(data.describe())

# 将特征字段分成3组
features_mean= list(data.columns[2:12])
features_se= list(data.columns[12:22])
features_worst=list(data.columns[22:32])

# 数据清洗
# ID列没有用，删除该列
data.drop("id",axis=1,inplace=True)
# 将B良性替换为0，M恶性替换为1
data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})

# 将肿瘤诊断结果可视化
sns.countplot(data['diagnosis'],label="Count")
plt.show()
# 用热力图呈现features_mean字段之间的相关性
corr = data[features_mean].corr()
plt.figure(figsize=(14,14))
# annot=True显示每个方格的数据
sns.heatmap(corr, annot=True)
plt.show()


# 特征选择（考虑到数据集的形式与结果的可解释性，手动选择代替PCA）
features_remain = ['radius_mean','texture_mean', 'smoothness_mean','compactness_mean','symmetry_mean', 'fractal_dimension_mean'] 

# 抽取30%的数据作为测试集，其余作为训练集
train, test = train_test_split(data, test_size = 0.3)# in this our main data is splitted into train and test
# 抽取特征选择的数值作为训练和测试数据
train_X = train[features_remain]
train_y=train['diagnosis']
test_X= test[features_remain]
test_y =test['diagnosis']

# 在使用核函数计算时，需要计算K（Xi，Xj）内积来更新a，因此采用Z-Score规范化数据，使数据在同一个量级上。
ss = StandardScaler()
train_X = ss.fit_transform(train_X)
test_X = ss.transform(test_X)

# 创建SVM分类器
model = svm.SVC()
# 用训练集做训练
model.fit(train_X,train_y)
# 用测试集做预测
prediction=model.predict(test_X)
print('准确率: ', metrics.accuracy_score(prediction,test_y))
#准确率:  0.9649122807017544
#SVM作为除神将网络以外最好的分类方法，取得较好的准确率不足为奇

#后续：
#高斯核函数SVC训练模型，6个特征变量，训练集准确率：96.0%，测试集准确率：92.4%
#高斯核函数SVC训练模型，10个特征变量，训练集准确率：98.7% ，测试集准确率：98.2%
#可以发现增加特征变量可以提高准确率，可能是因为模型维度变高，模型变得更加复杂。可以看出特征变量的选取很重要。


