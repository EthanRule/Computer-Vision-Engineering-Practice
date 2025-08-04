import pandas as pd

students = pd.DataFrame({"Name": ["Alice", "Bob", "Charlie"], "Age": [24, 27, 22], "Score": [85, 90, 88]})
print(students)

greater_than_85 = students.loc[students["Score"] > 85, "Name"]
print(greater_than_85)