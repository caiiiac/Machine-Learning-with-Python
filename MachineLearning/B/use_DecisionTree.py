# 导入sklearn内嵌的鸢尾花数据集
from sklearn.datasets import load_iris

# 导入决策树分类器, 同时导入计算交叉验证值的函数 cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# 使用默认参数,创建一颗基于基尼系数的决策树, 并将该决策树分类器赋值给变量 clf
clf = DecisionTreeClassifier()

# 将鸢尾花数据赋值给变量 iris
iris = load_iris()

# 我们将决策树分类器作为待评估的模型, iris.data鸢尾花数据作为特征, iris.target鸢尾花分类作为目标结果, 通过设定cv为10, 使用10折交叉验证
# 得到最终的交叉验证得分
print(cross_val_score(clf, iris.data, iris.target, cv=10))