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

# import pandas as pd

# titanic = pd.read_csv("data/titanic.csv")
# print(titanic.head())

# ages = titanic["Age"]
# print(ages.head())
# print(type(titanic["Age"]))

# print(titanic["Age"].shape)

# age_sex = titanic[["Age", "Sex"]]
# print(age_sex.head())
# print(type(titanic[["Age", "Sex"]]))

# print(titanic[["Age", "Sex"]].shape)

# above_35 = titanic[titanic["Age"] > 35]
# print(above_35.head())

# print(above_35.shape)

# class_23 = titanic[titanic["Pclass"].isin([2, 3])]
# print(class_23.head())

# class_23 = titanic[(titanic["Pclass"] == 2) | (titanic["Pclass"] == 3)]
# print(class_23.head())

# age_no_na = titanic[titanic["Age"].notna()]
# print(age_no_na.head())
# print(age_no_na.shape)
# cabin_no_na = titanic[titanic["Cabin"].notna()]
# print(cabin_no_na.head())

# adult_names = titanic.loc[titanic["Age"] > 35, "Name"]
# print(adult_names.head())

# print(titanic.iloc[9:25, 2:5])

# titanic.iloc[0:3, 3] = "anonymous"
# print(titanic.head())

import pandas as pd

import matplotlib.pyplot as plt

air_quality = pd.read_csv("data/air_quality_no2.csv", index_col=0, parse_dates=True)

print(air_quality.head())

fig, axs = plt.subplots(figsize=(12, 4))
air_quality.plot.area(ax=axs)
axs.set_ylabel("NO$_2$ concentration")
fig.savefig("no2_concentrations.png")
plt.show()