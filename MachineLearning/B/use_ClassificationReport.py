import pandas as pd
import numpy as np

# 从sklearn库中导入预处理模块 Imputer
from sklearn.preprocessing import Imputer
# 导入自动生成训练集和测试集的模块 train_test_split
from sklearn.cross_validation import train_test_split
# 导入预测结果评估模块 classification_report
from sklearn.metrics import classification_report

# 导入K近邻分类器 决策树分类器 高斯朴素贝叶斯
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# 读取特征文件列表和标签文件中的内容,归并后返回
def load_datasets(feature_paths, label_paths):
    feature = np.ndarray(shape=(0,41))
    label = np.ndarray(shape=(0,1))
    for file in feature_paths:
        # 使用pandas库的read_table函数读取一个特征文件内容
        # 指定分隔符为逗号 缺失值为问号 文件中不包含表头行
        df = pd.read_table(file, delimiter=',', na_values='?', header=None)
        # 使用Imputer函数,通过设定strategy参数为'mean'
        # 使用平均值对缺失数据补全,fit()函数用于训练预处理器,
        # transform()函数用于生成预处理结果
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        imp.fit(df)
        df = imp.transform(df)
        # 将预处理后的数据加入feature,依次遍历完所有特征文件
        feature = np.concatenate((feature, df))
     
    for file in label_paths:
        # 同上
        df = pd.read_table(file, header=None)
        # 标签文件没有缺失值,所以直接将读取到的新数据加入label集合
        label = np.concatenate((label, df))
         
    label = np.ravel(label)
    # 将特征集合feature与标签集合label返回
    return feature, label
 
if __name__ == '__main__':
    ''' 数据路径 '''
    featurePaths = ['A/A.feature','B/B.feature','C/C.feature','D/D.feature','E/E.feature']
    labelPaths = ['A/A.label','B/B.label','C/C.label','D/D.label','E/E.label']
    ''' 读入数据  '''
    # 使用分片方法,将数据路径中前4个数据作为训练集传入load_datasets()
    # 得到训练集的特征x_train,训练集的标签y_train
    x_train,y_train = load_datasets(featurePaths[:4],labelPaths[:4])
    # 将最后一个数据作为测试集,传入load_datasets()
    # 得到测试集的特征x_test,训练集的标签y_test
    x_test,y_test = load_datasets(featurePaths[4:],labelPaths[4:])
    # 使用train_test_split()函数,通过设置测试集比例test_size为0,将数据随机打乱,便于后续分类器的初始化和训练
    x_train, x_, y_train, y_ = train_test_split(x_train, y_train, test_size = 0.0)
    
    # 创建K近邻分类器,并将训练集x_train和y_train传入fit()函数进行训练
    print('Start training knn')
    knn = KNeighborsClassifier().fit(x_train, y_train)
    print('Training done')
    # 使用测试集x_test,进行分类器预测,得到分类结果
    answer_knn = knn.predict(x_test)
    print('Prediction done')
    
    # 创建决策树分类器,并将训练集x_train和y_train传入fit()函数进行训练
    print('Start training DT')
    dt = DecisionTreeClassifier().fit(x_train, y_train)
    print('Training done')
    answer_dt = dt.predict(x_test)
    print('Prediction done')
    
    # 创建贝叶斯分类器,并将训练集x_train和y_train传入fit()函数进行训练
    print('Start training Bayes')
    gnb = GaussianNB().fit(x_train, y_train)
    print('Training done')
    answer_gnb = gnb.predict(x_test)
    print('Prediction done')
    
    # 使用classification_report()函数对分类结果,从精确率precision 召回率recall f1值f1-sorce和支持度support四个维度进行衡量
    # 分别对三个分类结果进行输出
    print('\n\nThe classification report for knn:')
    print(classification_report(y_test, answer_knn))
    print('\n\nThe classification report for DT:')
    print(classification_report(y_test, answer_dt))
    print('\n\nThe classification report for Bayes:')
    print(classification_report(y_test, answer_gnb))