import numpy as np

A = np.arange(1, 7).reshape(2, 3)

B = np.arange(7, 13).reshape(3, 2)

C = A @ B
print(C)
A_transpose = np.transpose(A)
print(A_transpose)