# 使用import语句导入K近邻分类器
from sklearn.neighbors import KNeighborsClassifier

# 创建一组数据x 和它对应的标签y
x=[[0],[1],[2],[3]]
y=[0,0,1,1]

# 参数n_neighbors为3,即 使用最近的3个邻居作为分类的依据, 其它参数保持默认值, 将创建好的实例赋值给变量neigh
neigh=KNeighborsClassifier(n_neighbors=3)
# 调用fit()函数, 将训练数据x 和标签y 送入分类器进行学习
neigh.fit(x,y)

# 调用predict()函数, 对未知分类样本[1,1]分类, 可以直接并将需要分类的数据造为数组形式作为参数传入, 得到分类标签作为返回值
print(neigh.predict([[1.1]]))

# 样例输出值是 0, 表示近邻分类器通过计算样本[1,1]与训练数据的距离, 取0,1,2,这三个邻居作为依据, 根据'投票法'最终将样本分为类别0