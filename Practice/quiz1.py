import numpy as np

# 1-5
arr = np.random.randint(10, 25, (4, 4))
np.flip(arr, axis=1) # horizontal
np.flip(arr, axis=0) # vertical

arr.clip(12, 20)

# normalize?

# 
arr = np.ones((64, 64, 3), dtype=np.uint8)
arr[:, :, 1] = 0

# 6-10
import pandas as pd
data = {
    "image_id": [1, 2, 3, 4, 5, 6, 7, 8],
    "class": ["A", "B", "C", "A", "B", "C", "A", "B"],
    "confidence": [1, 0.5, 0.25, 0.75, 0.5, 0.75, 0.5, 0]
}

df = pd.DataFrame(data)

means = df.groupby("class")["confidence"].mean()
print(means)

filtered_df = df[df["confidence"] > 0.7]
print(filtered_df)

#filtered_df["is_high_confidence"] = df["confidence"] > 0.7?

pivot = pd.pivot_table(df, index="image_id", values="confidence", aggfunc="sum")
print(pivot)

# 11-13
import matplotlib.pyplot as plt
vals = np.random.normal(0, 1, 1000)
#plt.hist(vals, bins=40)
#plt.show()

points = np.random.random(50)

plt.scatter(points)
