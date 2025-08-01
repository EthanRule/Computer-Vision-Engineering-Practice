import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)

monthly_sales = rng.integers(50, 200, 36).reshape(12, 3)
print(monthly_sales)

df = pd.DataFrame()

df["Month"] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
df["Product A"] = monthly_sales[:, 0]
df["Product B"] = monthly_sales[:, 1]
df["Product C"] = monthly_sales[:, 2]
df["Total Sales"] = df["Product A"] + df["Product B"] + df["Product C"]
print(df)

highest_sales_month = df.loc[df["Total Sales"].idxmax(), "Month"]
print(highest_sales_month)

A = df["Product A"].sum() # is there a better way to do this just in pandas?
B = df["Product B"].sum()
C = df["Product C"].sum()
best_product = df[["Product A", "Product B", "Product C"]].sum().idxmax()


print(df.loc[df["Product B"].idxmin(), "Month"])

fig, ax = plt.subplots(3)
ax[0].set_title("Monthly Sales (Line Chart)")
ax[0].plot(df["Month"], df["Product A"], label="Product A")
ax[0].plot(df["Month"], df["Product B"], label="Product B")
ax[0].plot(df["Month"], df["Product C"], label="Product C")
ax[0].grid()
best_month_idx = df["Total Sales"].idxmax()
ax[0].annotate("Best Month", xy=(best_month_idx, df["Total Sales"].max()),
                             xytext=(best_month_idx, df["Total Sales"].max()+50),
                             arrowprops=dict(facecolor='black'))
ax[1].set_title("Monthly Sales (Bar Chart)")
ax[1].bar(df["Month"], df["Total Sales"], label="Product A")
ax[2].set_title("Monthly Sales (Pie Chart)")
labels = ["Product A", "Product B", "Product C"]
ax[2].pie([A, B, C], labels=labels, autopct='%1.1f%%')
fig.tight_layout()


plt.show()

