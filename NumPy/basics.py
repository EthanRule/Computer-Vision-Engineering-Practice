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

arr = np.array([2, 1, 5, 3, 7, 4, 6, 8])
print(np.sort(arr))

a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

print(np.concatenate((a, b)))

x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6]])

print(np.concatenate((x, y), axis=0))

array_example = np.array([[[0, 1, 2, 3],
                           [4, 5, 6, 7]],
                          [[0, 1, 2, 3],
                           [4, 5, 6, 7]],
                          [[0 ,1 ,2, 3],
                           [4, 5, 6, 7]]])

print(array_example.ndim)

print(array_example.size)

print(array_example.shape)

a = np.arange(6)
print(a)

b = a.reshape(3, 2)

print(b)

print(np.reshape(a, shape=(1, 6), order='C'))
