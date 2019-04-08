import numpy as np
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

x = np.arange(-10,10, 0.1)
y = x**2 -3*x + 5

plt.plot(x,y)
plt.show()
