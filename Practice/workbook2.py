# PART A
# 1. 
# A list is a vector of elements in python thats mutable, and can also act as a stack or queue.
# A tuple is a restricted size data structure that can only hold n elements. Tuples are not
# mutable.
# Finally a numpy array is the same as a list, but is built using c/c++ libraries for speed
# optimizations. Usually numpy is prefered because it has many built in functions useful for 
# matrix creations, reshaping, and other matrix operations. It also has attributes like .ndim
# and .shape. Also NumPy allows a user to specify a datatype like u8 to u64 and i8 to i64 and so on.
# Also syntax wise NumPy is far more consise. NumPy arrays are mutable and highly performant.

# NumPy arrays are the primary data structure used in CV pipelines because an image of say
# 256 x 256 x 3 for a rgb 256x256 pixel image can be used to run transformations on every pixel like
# flipping vertically or horizontally or applying a gradient. 

# 2. axis=0 is the first dimention (col), and axis=1 is the second dimention (row)
# Example

import numpy as np

arr = np.arange(1, 10).reshape(3, 3)
print(arr)

print(np.flip(arr, axis=0)) # flips column values
print(np.flip(arr, axis=1)) # flips row values

# Broadcasing in NumPy is applying a transformation to all elements in a numpy array. 

# not sure how to broadcast a rgb mean subtraction

# 3.
# .loc will give allow the user to specify the column row name when searching for something.
# .iloc will give the row index when serching for something.

# merge() outputs a new dataframe while concat() concats one dataframe to an existing one

# You might use merge to combine two images into a new image and concat to extend an image given
# new data.

# imshow() is used for opencv when displaying an image, while plot is for plotting a line graph.
# cmap='gray' makes it so the imshow() uses only two channels

# PART B

# 5
image = np.random.randint(0, 255, (64 * 64 * 3)).reshape(64, 64, 3)
print(image)
image[:, :, 0] = 0
image_flipped = np.flip(image, axis=1)

# 6
import pandas as pd
df = pd.DataFrame({"filename": ["hello", "world", "hi"], "width": [1, 2, 3], "height": [4, 5, 6], "label": ["mouse", "dog", "cat"]})
df["aspect_ratio"] = df["width"] / df["height"]
print(df)
aspect_ratios = df[(df["aspect_ratio"] > 1.5) & (df["label"] == "cat")]
# how to sort?
df.sort_values(by="width", ascending=False)

# 7
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3)
gray_image = np.mean(image, axis=2).astype(np.uint8)
ax[0].imshow(image)
ax[1].imshow(image_flipped)
ax[2].hist(gray_image.ravel(), bins=20)

# not sure how to do this

ax[0].plot()

# 8. I dont see why this wouldent work.
image = np.random.randint(0, 255, (64, 64))
normalized = image / 255
print(normalized.min(), normalized.max())



# 9.
df = pd.DataFrame({"image_id": [1, 2, 3, 4], "xmin": [1, 2, 3, 4], "ymin": [4, 8, 9, 10], "xmax": [14, 22, 33, 41], "ymax": [91, 82, 73, 64], "confidence": [1, 0.5, 0.25, 0.75]})

df["width"] = df["xmax"] - df["xmin"]
df["height"] = df["ymax"] - df["ymin"]
df["area"] = df["width"] * df["height"]

df = df[df["confidence"] > 0.6] # assmuming we want confidence over 0.6 since under dont really matter
print(df)

means = df.groupby("image_id")["area"].mean() 
print(means)

# 10 idk

fix, ax = plt.subplots()
ax.imshow(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
rect1 = plt.Rectangle((10, 10), 40, 30, edgecolor='red', facecolor='none')
rect2 = plt.Rectangle((50, 60), 20, 20, edgecolor='blue', facecolor='none')

ax.add_patch(rect1)
ax.add_patch(rect2)
plt.show()


