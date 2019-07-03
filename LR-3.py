#coding = UTF-8
from sklearn.linear_model import LogisticRegression as Log
import numpy as np
from sklearn import  linear_model,datasets


iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

model = Log()
model.fit(X, y, sample_weight=None)
print('y:', y)
print('predict:', model.predict(X))