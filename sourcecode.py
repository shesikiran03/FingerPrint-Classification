#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 18:17:58 2018

@author: apple
"""

import numpy as np
import pandas as pd
from scipy.misc import imread
import glob
import re
trainf= glob.glob('*.bmp')
testf= glob.glob('*.bmp')
trainfimages=[]
for file_name in trainf :
    imgtrainf = imread(file_name, flatten= 1)
    idf = pd.DataFrame(imgtrainf)
    idf2 = idf.iloc[45:230,45:230]
    id1 = idf2.values.flatten()
    trainfimages.append(id1)
testfimages=[]
for file_name in testf :
    imgtestf = imread(file_name, flatten= 1)
    idf3 = pd.DataFrame(imgtestf)
    idf5 = idf3.iloc[45:230,45:230]
    id4 = idf5.values.flatten()
    testfimages.append(id4)    
xtrain_frame= pd.DataFrame(trainfimages)
xtest_frame = pd.DataFrame(testfimages) 

xtrain_frame =xtrain_frame/255
a = xtrain_frame.values.sum()
b = a/ (xtrain_frame.size)
xtrain_frame = xtrain_frame - b


xtest_frame =xtest_frame/255
c = xtest_frame.values.sum()
d = c/ (xtest_frame.size)
xtest_frame = xtest_frame - d


#from sklearn.preprocessing import LabelEncoder
#lb=LabelEncoder()
#xtrain_frame_encoded=xtrain_frame.apply(lb.fit_transform)
#xtest_frame_encoded=xtest_frame.apply(lb.fit_transform)




ytrain_frame = []
for x in trainf :
    y = x.split('_')
    ytrain_frame.append(int(y[0]))
    
 ytrain_frame = sorted(ytrain_frame)

ytest_frame = []
for p in testf :
    q = p.split('_')
    ytest_frame.append(int(q[0]))
    
 ytest_frame = sorted(ytest_frame)



from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
OneVsRestClassifier(LinearSVC(random_state=0))
from sklearn.model_selection import GridSearchCV
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc=svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
clf = GridSearchCV(svc, parameters)
clf.fit(xtrain_frame,ytrain_frame)
y_pred=clf.predict(xtest_frame)
from sklearn.metrics import accuracy_score
accuracy_score(ytest_frame,y_pred)

       






 
   
