import matplotlib.pyplot as plt
import numpy as np

x_values = np.arange(0, 5)
y_values = np.arange(0, 5)**2
print(x_values)
print(y_values)

plt.plot(x_values, y_values)
plt.title("Quadratic Growth")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()