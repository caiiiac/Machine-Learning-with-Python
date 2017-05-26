import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot

# 测试题
# 假设有如下八个点：（3,1）（3,2）（4,1）（4,2）（1,3）（1,4）（2,3）（2,4），使用KMeans算法对其进行聚类。假设初始聚类中心点分别为（0,4）和（3,3），则最终的聚类中为（____,____）和（____,____）。
# 注：答案数字以逗号分隔，如1.2,3.5,4.3,5.6

points = np.array([[3,1],[3,2],[4,1],[4,2],[1,3],[1,4],[2,3],[2,4]])
# pyplot.scatter(points[:,0],points[:,1])
# pyplot.show()

# 把数据点分组
clf = KMeans(n_clusters = 2)
clf.fit(points)

# 数据点的中心点
centers = clf.cluster_centers_
print(centers)


# 每个数据点所属分组
labels = clf.labels_
# print(labels)

for i in range(len(labels)):
    pyplot.scatter(points[i][0], points[i][1], c=('r' if labels[i] == 0 else 'b'))
    pyplot.scatter(centers[:,0],centers[:,1], marker='*', s=100)

# 预测
predict = [[1.5,1.5], [3.5,3.5]]
label = clf.predict(predict)
for i in range(len(label)):
    pyplot.scatter(predict[i][0], predict[i][1], c=('r' if label[i] == 0 else 'b'), marker='x')

pyplot.show()
