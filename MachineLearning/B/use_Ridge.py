import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

data = np.genfromtxt('data.txt')
plt.plot(data[:,4])

x = data[:,:4]
y = data[:,4]
poly = PolynomialFeatures(6)
X = poly.fit_transform(x)

train_set_x, test_set_x, train_set_y, test_set_y = train_test_split(X,y,test_size=0.3,random_state=0)
clf = Ridge(alpha=1.0,fit_intercept=True)
clf.fit(train_set_x,train_set_y)
clf.score(test_set_x,test_set_y)

start = 200
end = 300
y_pre = clf.predict(X)
time = np.arange(start,end)
plt.plot(time,y[start,end],'b',label='real')
plt.plot(time,y_pre[start,end],'r',label='predict')

plt.legend(loc='upper left')
plt.show()
