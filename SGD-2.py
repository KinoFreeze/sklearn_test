from sklearn.linear_model import SGDClassifier as SGD
import matplotlib.pyplot as plt
import numpy as np

x=[[0,0],[1,1],[2,2],[3,3]]
y=[0,1,2,3]
clf = SGD(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='hinge', max_iter=5, n_iter=None,
       n_jobs=1, penalty='l2', power_t=0.5, random_state=None,
       shuffle=True, tol=None, verbose=0, warm_start=False)
clf.fit(x,y)
print(clf.predict([[4,4]]))
print(clf.coef_)
print(clf.intercept_)
print(clf.decision_function([[2,2]]))