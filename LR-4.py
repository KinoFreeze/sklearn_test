from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt


x = np.array([6.19, 2.51, 7.29, 7.01, 5.7, 2.66, 3.98, 2.5, 9.1, 4.2]).reshape(-1, 1)
y = np.array([5.25, 2.83, 6.41, 6.71, 5.1, 4.23, 5.05, 1.98, 10.5, 6.3]).reshape(-1, 1)


model = linear_model.LinearRegression()

model.fit(x, y)

y_plot = model.predict(x)

print(model.coef_) ## 0.90045842

plt.scatter(x, y, color='red', label="dsaf", linewidth=2)
plt.plot(x, y_plot, color='green', label="asdf", linewidth=2)
plt.legend(loc='lower right')
plt.show()