import matplotlib.pyplot as plt
import numpy as np

y1 = np.arange(-5, 6)
y2 = np.arange(-5, 6)**2
y3 = np.arange(-5, 6)**3
x = np.arange(-5, 6)


plt.plot(x, y1, label="linear")
plt.plot(x, y2, label="quadratic")
plt.plot(x, y3, label="cubic")
plt.title("Linear, Quadratic, and Cubic Functions")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()