# 导入numpy库, 构造训练数据X , y
import numpy as np
X = np.array([[-1,-1], [-2,-1], [-3,-2], [1,1], [2,1], [3,2]])
y = np.array([1,1,1,2,2,2])

# 导入朴素贝叶斯分类器
from sklearn.naive_bayes import GaussianNB

# 使用默认参数, 创建一个高斯朴素贝叶斯分类器, 并将该分类器赋给变量clf
clf = GaussianNB(priors=None)

# 使用fit()进行训练, 使用predict()进行预测, 
clf.fit(X, y)
print(clf.predict([-0.8, -1]))