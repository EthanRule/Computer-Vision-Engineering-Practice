import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)

fig, ax = plt.subplots(3)
ax[0].plot(x, y1)
ax[0].set_title("Sine Wave")
ax[1].plot(x, y2)
ax[1].set_title("Cosine Wave")
ax[2].plot(x, y3)
ax[2].set_title("Tangent Wave")
ax[2].set_ylim(-10, 10)
plt.tight_layout()
plt.show()