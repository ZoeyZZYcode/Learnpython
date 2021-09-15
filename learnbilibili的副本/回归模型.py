#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing as fch
feature = fch().data
target = fch().target
x_train,x_test,y_train,y_test = train_test_split(feature,target,test_size=0.1,random_state=2020)
linner = LinearRegression()
linner.fit(x_train,y_train)
print(linner.coef_)
print(linner.intercept_)


#====评估指标======#
from sklearn.metrics import mean_squared_error
mean_squared_error(y_true=y_test,y_pred=linner.predict(x_test))
#交叉验证
from sklearn.model_selection import cross_val_score
cross_val_score(linner,x_train,y_train,cv=5,scoring='neg_mean_squared_error').mean()# absolute
#R=1- y-yhat/y-ymean#3way
from sklearn.metrics import r2_score
r2_score(y_test,linner.predict(x_test))
r2= linner.score(x_test,y_test)
cross_val_score(linner,x_test,y_test,cv=10,scoring='r2').mean()
import matplotlib.pyplot as plt
y_pred=linner.predict(x_test)
plt.plot(range(len(y_test)),sorted(y_test),c='b',label='y_true')
plt.plot(range(len(y_pred)),sorted(y_pred),c='r',label='y_pred')
plt.legend()
plt.show()
"""
#====多项式回归=====#
'''

from sklearn.preprocessing import PolynomialFeatures
PolynomialFeatures(include_bias=,interaction_only=,degree=).fit_transform()
'''

#====岭回归=====#
"""
from sklearn.linear_model import Ridge
alpha=0-1 or 1-10

"""
#======模型保存和加载======#
from sklearn.externals import joblib
joblib.dump(model,'xxx.m')#save
joblib.load('xx.m')# 加载
