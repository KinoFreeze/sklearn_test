﻿import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as Log

data = [
       [-0.017612, 14.053064, 0],
       [-1.395634, 4.662541, 1],
       [-0.752157, 6.538620, 0],
       [-1.322371, 7.152853, 0],
       [0.423363, 11.054677, 0],
       [0.406704, 7.067335, 1],
       [0.667394, 12.741452, 0],
       [-2.460150, 6.866805, 1],
       [0.569411, 9.548755, 0],
       [-0.026632, 10.427743, 0],
       [0.850433, 6.920334, 1],
       [1.347183, 13.175500, 0],
       [1.176813, 3.167020, 1],
       [-1.781871, 9.097953, 0],
       [-0.566606, 5.749003, 1],
       [0.931635, 1.589505, 1],
       [-0.024205, 6.151823, 1],
       [-0.036453, 2.690988, 1],
       [-0.196949, 0.444165, 1],
       [1.014459, 5.754399, 1]
       ]

dataMat = np.mat(data)
y=dataMat[:, 2]
b=np.ones(y.shape)
x=np.column_stack((b, dataMat[:, 0:2]))
x=np.mat(x)

model = Log()
model.fit(x, y)
print(model)

predicted = model.predict(x)
answer = model.predict_proba(x)
print('predicted:', predicted)
print(answer)
