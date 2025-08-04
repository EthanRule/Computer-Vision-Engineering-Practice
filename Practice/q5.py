import pandas as pd

products = pd.DataFrame({"Product": ["Apple", "Banana", "Orange", "Mango"], "Price": [2.5, 1.0, 1.5, 3.0], "Quantity": [5, 12, 8, 3]})

greater_than_five = products[products["Quantity"] > 5]
print(greater_than_five)
sorted_values = greater_than_five.sort_values(by="Price", ascending=False)
print(sorted_values)
