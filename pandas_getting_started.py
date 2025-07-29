# import pandas as pd

# df = pd.DataFrame(
#     {
#         "Name": [
#             "Braund, Mr. Owen Harris",
#             "Allen, Mr. William Henry",
#             "Bonnell, Miss. Elizabeth",
#         ],
#         "Age": [22, 35, 58],
#         "Sex": ["male", "male", "female"],
#     }
# )

# print(df)
# print(df["Age"])


# ages = pd.Series([22, 35, 58], name="Age")
# print(ages)

# print(df["Age"].max())
# print(ages.max())

# print(df.describe())

# import pandas as pd
# titanic = pd.read_csv("data/titanic.csv")
# print(titanic)
# print(titanic.head(8))
# print(titanic.tail(10))
# print(titanic.dtypes)

# # titanic.to_excel("titanic.xlsx", sheet_name="passengers", index=False)

# print(titanic.info())

import pandas as pd

