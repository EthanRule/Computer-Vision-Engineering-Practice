import numpy as np

#1 
array = np.arange(0, 25).reshape(5, 5)
print(array)
array[array % 2 == 0] = -1
print(array)

#2 
array = np.random.uniform(0, 10, size=100)
print(array)

array = array.astype(np.float32)
normalized = (array - array.min()) / (array.max() - array.min())
print(normalized)

# d:\Repos\Computer-Vision-Engineering-Practice\Practice\MOREPROBLEMS.py:18: RuntimeWarning: divide by zero encountered in divide
#   normalized = (array - array.min()) / (array.max() - array.max())
# d:\Repos\Computer-Vision-Engineering-Practice\Practice\MOREPROBLEMS.py:18: RuntimeWarning: invalid value encountered in divide
#   normalized = (array - array.min()) / (array.max() - array.max())
# [inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf
#  inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf
#  nan inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf
#  inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf
#  inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf
#  inf inf inf inf inf inf inf inf inf inf]

