# coding = UTF-8
from sklearn.linear_model import SGDClassifier as SGD
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import numpy as np

X,y = make_blobs(n_samples=50,centers=2,random_state=0,cluster_std=0.6)
clf = SGD(loss='hinge',alpha=0.01,max_iter=200,fit_intercept=True)
clf.fit(X,y)
print("回归系数：",clf.coef_)
print("偏差",clf.intercept_)
print("##################")
print(X.shape)
print(y.shape)
