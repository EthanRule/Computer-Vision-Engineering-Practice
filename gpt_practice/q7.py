import numpy as np

a = np.arange(1, 4)
b = np.arange(4, 7)

vertical_stack = np.vstack((a, b))
print(vertical_stack)
horizontal_stack = np.hstack((a, b))
print(horizontal_stack)