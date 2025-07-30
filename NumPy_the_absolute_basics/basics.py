import numpy as np

a = np.arange(1, 7)
print(a)

b = a[3:]
print(b)
b[0] = 40
print(a)

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(a)
print(a.ndim)
print(a.shape)
print(a.size)
print(a.dtype)

print(np.zeros(2))
print(np.ones(2))

print(np.empty(2))

print(np.arange(4))

print(np.arange(2, 9, 2))

print(np.linspace(0, 10, num=5))

x = np.ones(2, dtype=np.uint32)
print(x)
