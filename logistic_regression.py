# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 15:21:48 2020

@author: liushang
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import sys
inputf=sys.argv[1]
outputf=sys.argv[2]
#preprocessing data
data=pd.read_csv(inputf,sep='\t',index_col=0)
#get train and test data as input of model
factor=data.columns.tolist()[:-1]
print(factor)
X=data[factor]
#X['intercept']=[1 for i in range(X.shape[0])]
Y=data['label']
Y=[0 if i==1 else 1 for i in Y]
x_train=np.array(X.iloc[3000:,:])
y_train=np.array(Y[3000:])
x_test=np.array(X.iloc[:3000,:])
y_test=np.array(Y[:3000])
np.save(outputf+'_test.npy',y_test)
'''
lr=sm.Logit(y_train,x_train)
result=lr.fit()
print (result.summary())
'''
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(solver='liblinear',max_iter=1000,penalty='l1')
result=clf.fit(x_train,y_train)
#predict test data
from sklearn.metrics import accuracy_score,recall_score,precision_score
pred_y=result.predict(x_test)
pred_y=np.array([1 if i>0.5 else 0 for i in pred_y])
np.save(outputf+'_pred.npy',pred_y)
#evaluate
print(result.coef_)
print(result.intercept_)
print(accuracy_score(y_test,pred_y))
print(recall_score(y_test,pred_y))
print(precision_score(y_test,pred_y))
f1_score=2/((1/recall_score(y_test,pred_y))+(1/precision_score(y_test,pred_y)))
print(f1_score)
