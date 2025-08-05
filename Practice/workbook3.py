import numpy as np

image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
image[:, :, 1] = 255
print(image)
vertical_flip = np.flip(image, axis=1)

import pandas as pd
df = pd.DataFrame({
    "filename": ["a.jpg","b.jpg","c.jpg","d.jpg"],
    "width": [100, 300, 150, 400],
    "height": [200, 100, 150, 100],
    "label": ["cat","dog","cat","cat"]
})
df["aspect_ratio"] = df["width"] / df["height"]

cat_images = df[(df["aspect_ratio"] > 1/2) & (df["label"] == "cat")]
cat_images.sort_values(by="width", ascending=False)
print(cat_images)