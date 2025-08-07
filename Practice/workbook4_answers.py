import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1
arr = np.arange(1, 28).reshape(3, 3, 3)
print(arr.flatten())
# [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27]

# 2
data = {
    "store": ["A", "A", "B", "B", "C", "C"],
    "product": ["apple", "banana", "pear", "apple", "banana", "pear"],
    "sales": [10, 15, 5, 16, 3, 24]
}

df = pd.DataFrame(data)

pivot = pd.pivot_table(df, index="store", columns="product", values="sales", aggfunc="sum")
print(pivot)

# 3 using same dataframe
data = {
    "height": [150, 160, 170, 180, 190],
    "weight": [50, 60, 70, 80, 90],
    "age": [25, 30, 35, 40, 45]
}

df = pd.DataFrame(data)

corr_matrix = df.corr()
print(corr_matrix)

# 4
arr = np.array([5, 10, 15, 25, 30])
clipped = np.clip(arr, 10, 20)
print(clipped)

# 5

x = np.arange(1, 256)
y = np.random.randint(0, 255, 255)

# plt.title("Pixel colors")
# plt.scatter(x, y)
# plt.xlabel("pixel")
# plt.xlabel("color")
# plt.show()

# 6

# Z-Score Formula
#     x - μ
# z = -----
#       σ


arr = np.arange(0, 256)
mean = arr.mean()
std = arr.std()

z_scores = (arr - mean) / std
print(z_scores)

# 7
# y = np.random.normal(loc=0, scale=1, size=1000)
# plt.hist(y, bins=30, color='skyblue', edgecolor='black')
# plt.title("Normal Distribution Histogram")
# plt.xlabel("Value")
# plt.ylabel("Frequency")
# plt.show()

# 8
arr = np.random.random(100)
print(arr)
std = arr.std()
print(std)

# 9
rgb = np.zeros((256, 256, 3), dtype=np.uint8)
x = np.arange(1, 257)

fig, ax = plt.subplots(1, 3)
ax[0].set_title("red")
ax[1].set_title("green")
ax[2].set_title("blue")
ax[0].imshow(rgb[:, :, 0], cmap='Reds')
ax[1].imshow(rgb[:, :, 1], cmap='Greens')
ax[2].imshow(rgb[:, :, 2], cmap='Blues')
plt.show()