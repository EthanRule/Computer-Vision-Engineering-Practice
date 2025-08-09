import numpy as np
# c = a + b =
# [[1+1  2+2  3+3]
#  [4+4  5+5  6+6]
#  [7+7  8+8  9+9]]

# = [[ 2  4  6]
#    [ 8 10 12]
#    [14 16 18]]

def matrix_add(A, B):
    if A.shape != B.shape:
        raise ValueError("Matrices must have same shape to be added.")
    return A + B

#           [1 2 3]       [1 2 3]
# a =       [4 5 6]   b = [4 5 6]
#           [7 8 9]       [7 8 9]

# a[0] = [1 2 3]
# b[:,0] = [1 4 7]

# d[0,0] = 1*1 + 2*4 + 3*7 = 1 + 8 + 21 = 30

# d = a @ b =
# [
#   [1*1 + 2*4 + 3*7,  1*2 + 2*5 + 3*8,  1*3 + 2*6 + 3*9],
#   [4*1 + 5*4 + 6*7,  4*2 + 5*5 + 6*8,  4*3 + 5*6 + 6*9],
#   [7*1 + 8*4 + 9*7,  7*2 + 8*5 + 9*8,  7*3 + 8*6 + 9*9]
# ]

# = [
#   [ 30,  36,  42],
#   [ 66,  81,  96],
#   [102, 126, 150]
# ]

def matrix_multiply(A, B):
    # check if matrices are compatible for multiplication
    if A.shape[1] != B.shape[0]:
        raise ValueError("Matrices must have same shape.")
    return A @ B


a = np.arange(1, 10).reshape(3, 3)
b = np.arange(1, 10).reshape(3, 3)

c = matrix_add(a, b)
d = matrix_multiply(a, b)
print(c)
print(d)

# Dot product (can only be applied to 1d equilength matrices)

def dot_product(x, y):
    if len(x) != len(y):
        raise ValueError("x and y not same length")
    else:
        return x @ y
    
x = np.arange(1, 10)
y = np.arange(1, 10)

print(dot_product(x, y))

# Given a 2x2 matrix M and a vector V, write a function to apply lin transformation M ->V
# this is the same thing as matrix mult, but a special case where we use a single vect
def apply_transformation(M, v):
    M = np.array(M)
    v = np.array(v)

    if M.shape[1] != v.shape[0]:
        raise ValueError("Matrix and vector dim do not align")

    return M @ v

# A = [[a, b], [c, d]]
# det(A) = ad - bc
# so only if the det(A) != 0, the matrix is invertible
def determinant(matrix):
    if matrix.shape != (2, 2):
        raise ValueError("dims not 2x2")
    return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]

matrix = np.array([[1, 2], [3, 4]])
print(determinant(matrix))

def inverse_matrix(matrix):
    if determinant(matrix) == 0:
        raise ValueError("matrix not invertible")
    return np.linalg.inv(matrix)

print(inverse_matrix(matrix))

def eigen(matrix):
    if matrix.shape != (3, 3):
        raise ValueError("matrix not 3x3")
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    return (eigenvalues, eigenvectors)

matrix = np.arange(1, 10).reshape(3, 3)
eigens = eigen(matrix)

# def valid_eigens(matrix, eigens):