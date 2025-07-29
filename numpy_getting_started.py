# import numpy as np
# a = np.array([2, 3, 4])
# print(a)

# print(a.dtype)

# b = np.array([1.2, 3.5, 5.1])
# print(b.dtype)

# c = np.array([[1.5, 2, 3], [4, 5, 6]])
# print(c)

# d = np.array([[1, 2], [3, 4]], dtype=complex)

# print(np.zeros((3, 4)))

# print(np.ones((2, 3, 4), dtype=np.int16))

# print(np.empty((2, 3)))

# print(np.arange(10, 30, 5))

# print(np.arange(0, 2, 0.3))

# print(np.linspace(0, 2, 9))

# x = np.linspace(0, 2 * np.pi, 100)
# f = np.sin(x)
# print(f)

# print(np.arange(6))
# print(np.arange(12).reshape(4, 3))

# print(np.arange(24).reshape(2, 3, 4))

# g = np.array([20, 30, 40, 50])
# h = np.arange(4)
# c = g - h
# print(c)

# print(h**2)

# print(10 * np.sin(g))

# print(g < 35)

# A = np.array([[1, 1], [0, 1]])
# B = np.array([[2, 0], [3, 4]])

# print(A * B)

# print(A @ B)

# print(A.dot(B))

# rg = np.random.default_rng(1)
# Aa = np.ones((2, 3), dtype=int)
# Bb = rg.random((2, 3))
# Aa *= 3
# print(Aa)

# Bb += Aa
# print(Bb)

# Aa += Bb
import numpy as np
rg = np.random.default_rng(1)
from numpy import newaxis

# a = np.ones(3, dtype=np.int32)
# b = np.linspace(0, np.pi, 3)
# print(b.dtype)

# c = a + b
# print(c)

# print(c.dtype)

# d = np.exp(c * 1j)
# print(d)

# print(d.dtype)

# a = rg.random((2, 3))
# print(a)

# print(a.sum())
# print(a.min())
# print(a.max())

# b = np.arange(12).reshape(3, 4)
# print(b)

# print(b.sum(axis=0))

# print(b.min(axis=1))

# print(b.cumsum(axis=1))

# B = np.arange(3)
# print(B)
# print(np.exp(B))
# print(np.sqrt(B))
# C = np.array([2., -1., 4.])
# print(np.add(B, C))

# a = np.arange(10) ** 3
# print(a)

# print(a[2])

# print(a[2:5])

# a[:6:2] = 1000

# print(a)

# print(a[::-1])

# for num in a:
#     print(num**(1/3.))

# def f(x, y):
#     return 10 * x + y

# b = np.fromfunction(f, (5, 4), dtype=int)
# print(b)

# print(b[2, 3])

# print(b[0:5, 1])

# print(b[:, 1])

# print(b[1:3, :])

# print(b[-1])

# c = np.array([[[0, 1, 2],
#               [10, 12, 13]],
#              [[100, 101, 102],
#               [110, 112, 113]]])
# print(c.shape)

# print(c[1, ...])

# print(c[..., 2])

# for row in b:
#     print(row)

# for element in b.flat:
#     print(element)

# a = np.floor(10 * rg.random((3, 4)))
# print(a)

# print(a.shape)

# print(a.ravel())
# print(a.reshape(6, 2))

# print(a.T)
# print(a.T.shape)
# print(a.shape)

# print(a)
# a.resize((2, 6))
# print(a)

# print(a.reshape(3, -1))

a = np.floor(10 * rg.random((2, 2)))
print(a)

b = np.floor(10 * rg.random((2, 2)))
print(b)

print(np.vstack((a, b)))
print(np.hstack((a, b)))

print(np.column_stack((a, b)))
a = np.array([4., 2.])
b = np.array([3., 8.])
print(np.column_stack((a, b)))
print(np.hstack((a, b)))
print(a[:, newaxis])
print(np.column_stack((a[:, newaxis], b[:, newaxis])))
print(np.hstack((a[:, newaxis], b[:, newaxis])))

print(np.r_[1:4, 0, 4])

a = np.floor(10 * rg.random((2, 12)))
print(a)
