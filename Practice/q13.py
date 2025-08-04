import numpy as np

rng = np.random.default_rng()
arr = rng.integers(1, 20, 36).reshape(6, 6)
print(arr[1])
print(arr[4])
evens = arr[arr % 2 == 0]
print(evens)

arr[arr > 10] = 10
print(arr)

import pandas as pd

data = {
    "Product": ["Apple", "Banana", "Mango", "Orange", "Apple", "Banana", "Mango", "Orange"],
    "Month": ["Jan", "Jan", "Jan", "Jan", "Feb", "Feb", "Feb", "Feb"],
    "Sales": [100, 120, 90, 80, 150, 160, 120, 130]
}
df = pd.DataFrame(data)

print(df.groupby("Product")["Sales"].sum())
print(df.groupby("Month")["Sales"].sum())
df["Rank"] = df.groupby("Month")["Product"].rank(ascending=False, method="dense")
print(df)

import matplotlib.pyplot as plt

x = np.linspace(-np.pi, np.pi, 100)
sinx = np.sin(x)
cosx = np.cos(x)
tanx = np.tan(x)
tanx[np.abs(tanx) > 10] = np.nan

plt.plot(x, sinx, label="sinx")
plt.plot(x, cosx, label="cosx") 
plt.plot(x, tanx, label="tanx")
plt.title("sinx, cosx, tanx")
plt.grid()
plt.ylabel("Y")
plt.xlabel("X")
plt.legend()
plt.show()