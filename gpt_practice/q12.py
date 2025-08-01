import numpy as np

array = np.arange(1, 21).reshape(4, 5)
print(array)

second_row = array[1]
print(second_row)
last_column = array[:, -1]
print(last_column)

sub_matrix = array[1:3, 1:3]
print(sub_matrix)

import pandas as pd

sales_data = pd.DataFrame({"Product": ["Apple", "Banana", "Mango", "Orange"],
                           "Price": [2.5, 1.0, 3.0, 1.5],
                           "Quantity": [10, 20, 5, 15]})

sales_data["Total"] = sales_data["Price"] * sales_data["Quantity"]

total_sales_greater_than_20 = sales_data[sales_data["Total"] > 20]
print(total_sales_greater_than_20)

highest_total_sale = sales_data.loc[sales_data["Total"].idxmax(), "Product"]
print(highest_total_sale)

import matplotlib.pyplot as plt

x = np.arange(-5, 6)

y1 = x ** 2
y2 = x ** 3
y3 = np.sin(x)

plt.title("Quadratic, Cubic, Sine")
plt.plot(x, y1, label="Quadratic")
plt.plot(x, y2, label="Cubic")
plt.plot(x, y3, label="Sine")
plt.ylabel("Y")
plt.xlabel("X")
plt.grid(True)
plt.legend()
plt.show()