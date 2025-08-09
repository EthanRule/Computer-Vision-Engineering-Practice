import numpy as np

P = np.arange(1, 10).reshape(3, 3)
print(P)
R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
print(R)

P1 = P @ R.T
print(P1)