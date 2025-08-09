import numpy as np

A = 0.3
B = 0.4

def intersection(A, B):
    return A * B

def union(A, B):
    return A + B - intersection(A, B)

print(intersection(A, B)) # 0.12
print(union(A, B)) # 0.7

A = 0.01
B = 0.5
BA = 0.9

# Find A | B

def bayes_theorem(A, B, BA): # Determins A given B.
    return ((BA) * A) / B

print(bayes_theorem(A, B, BA)) # 0.018


numbers = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
def mean(numbers): # average
    return np.mean(numbers)

def variance(numbers): # how much values differ from the mean.
    return np.var(numbers)

def std(numbers): # how much variation or dispersion of a set of values around its mean. high std means values further from mean.
    return np.std(numbers)

print(mean(numbers)) # 5.5
print(variance(numbers)) # 8.25
print(std(numbers)) # 2.87