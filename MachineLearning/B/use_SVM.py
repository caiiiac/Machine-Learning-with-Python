# 相关包介绍
# pandas:用来加载csv数据的工具包
# numpy:支持高级大量的维度数据与矩阵运算,也针对数组运算提供大量的数学函数库
# svm: 算法
# cross_validation:交叉验证
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import cross_validation

# pd:pandas包的实例参数
# read_csv:参数一:数据源.encoding:编码格式.parse_dates:第n列解析为日期.index_col:用作索引的列编号
# sort_index:参数一:按0列排,ascending(true)升序,inplace:排序后是否覆盖原数据
data=pd.read_csv('stock/000777.csv',encoding='gbk',parse_dates=[0],index_col=0)
data.sort_index(0,ascending=True,inplace=True)
 
# dayfeature:选取150天的数据
# featurenum:选取5个特征*天数
# x:记录150天的5个特征值
# y:记录涨或者跌
# data.shape[0]-dayfeature:因为我们要用150天数据做训练,对于条目为200条的数据,只有50条数据有前150天的数据来训练的,所以训练集的大小就是200-150
# 对于每一条数据,他的特征是前150天的甩有特征数据,即150*5,+1是将当天的开盘价引入作为一条特征数据
dayfeature=150
featurenum=5*dayfeature
x=np.zeros((data.shape[0]-dayfeature,featurenum+1))
y=np.zeros((data.shape[0]-dayfeature))
 
for i in range(0,data.shape[0]-dayfeature):
    x[i,0:featurenum]=np.array(data[i:i+dayfeature] \
          [[u'收盘价',u'最高价',u'最低价',u'开盘价',u'成交量']]).reshape((1,featurenum))
    x[i,featurenum]=data.ix[i+dayfeature][u'开盘价']
 
for i in range(0,data.shape[0]-dayfeature):
    if data.ix[i+dayfeature][u'收盘价']>=data.ix[i+dayfeature][u'开盘价']:
        y[i]=1
    else:
        y[i]=0          
 
# 调用svm函数,并设置kernel参数,默认是rbf,其它:'linear','poly','sigmoid'
clf=svm.SVC(kernel='rbf')
result = []
for i in range(5):
    # x和y的验证集和测试集,切分80-20%的测试集
    x_train, x_test, y_train, y_test = \
                cross_validation.train_test_split(x, y, test_size = 0.2)
    # 训练数据进行训练
    clf.fit(x_train, y_train)
    # 将预测数据和测试集的验证数据比对
    result.append(np.mean(y_test == clf.predict(x_test)))
print("svm classifier accuacy:")
print(result)