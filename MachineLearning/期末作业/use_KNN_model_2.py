import numpy as np
from sklearn import neighbors

# fileName: 文件名字
# isTest: =True为测试数据 label直接设0
def readDataSet(fileName, isTest):
	fr = open(fileName,encoding='utf-8')
	lines = fr.readlines()

	numLabels = len(lines) 
	labels = np.zeros(numLabels)
	dateSet = np.zeros([numLabels,54],int)

	# 逐行读取数据到dataSet,labels
	# 总共55列,前54列是样本特征,最后一列是样本类别(label)
	for i in range(numLabels):
		line = lines[i]
		label = 0
		if isTest:
			label = 0
		else:
			label = line.split(' ')[54]
		labels[i] = label

		dates = np.zeros(54)
		for j in range(53):
			dates[j] = line.split(' ')[j]
		dateSet[i] = dates

	fr.close()
	return dateSet,labels


# read dataSet
train_dataSet,train_labels = readDataSet('data_train.txt', False)


knn = neighbors.KNeighborsClassifier(algorithm='kd_tree', n_neighbors=3)
print(knn)
knn.fit(train_dataSet,train_labels)

# read test_dataSet
test_dataSet,test_labels = readDataSet('data_test.txt', True)
res = knn.predict(test_dataSet)

# 输出预测结果,保存到model_1.txt
mf = open('model_2.txt','w')

for i in range(len(res)):
	mf.write(str(res[i]) + '\n')
mf.close()
