from sklearn.linear_model import LinearRegression as lr
import matplotlib.pyplot as plt
import numpy as np

x = np.array([3.6,4.5,2.6,4.9,2.5,3.5]).reshape(-1,1)
y = np.array([9.7,8.1,7.6,8.6,9.0,7.8]).reshape(-1,1)

model = lr()

model.fit(x,y)

y_plot = model.predict(x)

print(model.coef_)

plt.scatter(x,y,color='red',label= "sample data",linewidth=2)
plt.plot(x,y_plot,color='green',label="fitting line",linewidth=2)
plt.legend(loc='lower right')
plt.show()