import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

monthly_sales = pd.DataFrame({"Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                              "Product A": [100, 120, 140, 114, 150, 130, 150, 180, 193, 109, 140, 200],
                              "Product B": [80, 90, 95, 85, 92, 100, 105, 140, 110, 120, 140, 150],
                              "Product C": [50, 65, 90, 120, 110, 105, 115, 95, 64, 52, 55, 67]})

product_A_total_sales = monthly_sales["Product A"].sum()
product_B_total_sales = monthly_sales["Product B"].sum()
product_C_total_sales = monthly_sales["Product C"].sum()
print(product_A_total_sales)
print(product_B_total_sales)
print(product_C_total_sales)

monthly_sales["Total Sales"] = monthly_sales[["Product A", "Product B", "Product C"]].sum(axis=1)
best_month = monthly_sales.loc[monthly_sales["Total Sales"].idxmax(), "Month"]
print(best_month)

plt.plot(monthly_sales["Month"], monthly_sales["Product A"], label="Product A")
plt.plot(monthly_sales["Month"], monthly_sales["Product B"], label="Product B")
plt.plot(monthly_sales["Month"], monthly_sales["Product C"], label="Product C")
plt.legend()
plt.xticks(rotation=45)
plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Monthly Sales Trends")
plt.show()