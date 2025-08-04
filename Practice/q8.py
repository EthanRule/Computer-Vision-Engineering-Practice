import pandas as pd

departments = pd.DataFrame({"Department":
                            ["HR", "HR", "IT", "IT", "Sales"],
                            "Employee":
                            ["Alice", "Bob", "Charlie", "David", "Eva"],
                            "Salary":
                            [50000, 52000, 60000, 65000, 55000]})

# Goup the data by Department

print(departments.groupby("Department", as_index=False)["Salary"].mean())