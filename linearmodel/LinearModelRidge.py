# coding=UTF-8
# 广义线性回归-岭回归
import warnings
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
from sklearn import linear_model
reg = linear_model.Ridge (alpha = .5)
dataX=[[0, 0], [0, 0], [1, 1]]
dataY=[0,0.1,1]
print(reg.fit ([[0, 0], [0, 0], [1, 1]], [0, .1, 1]))
#Ridge(alpha=0.5, copy_X=True, fit_intercept=True, max_iter=None,
#normalize=False, random_state=None, solver='auto', tol=0.001)
print(reg.coef_)
#array([ 0.34545455,  0.34545455])
print(reg.intercept_)
#正常情况dataX应该是test data
diabetes_y_pred=reg.predict(dataX)


#plt.scatter(dataX, dataY, color='black', linewidth=3)
#line
plt.plot(dataX, diabetes_y_pred, color='blue', linewidth=3)

plt.show()
#0.13636...