# 100 NumPy, Pandas, Matplotlib problems
import random

# Define a pool of problem templates
numpy_problems = [
    "Create a NumPy array of shape (5, 5) with random integers between 0 and 10.",
    "Create a 3D NumPy array and reverse its second axis.",
    "Generate a 1D NumPy array of 50 values linearly spaced between 0 and 1.",
    "Create a 4x4 identity matrix and multiply it by 5.",
    "Flatten a 3x3x3 NumPy array to 1D.",
    "Reshape a 1D array of 36 elements into shape (3, 3, 4).",
    "Replace all negative values in a NumPy array with 0.",
    "Broadcast a 1D array across a 2D array and add them together.",
    "Calculate the standard deviation of a random NumPy array.",
    "Generate a (5,5) array and set the border elements to 1, and the inner to 0.",
    "Implement z-score normalization on a NumPy array.",
    "Compute the row-wise sum of a 2D array.",
    "Find the most frequent element in a NumPy array.",
    "Clip all values in a NumPy array between 10 and 20.",
    "Generate a checkerboard pattern using NumPy.",
]

pandas_problems = [
    "Create a Pandas DataFrame with 3 columns: name, age, and score.",
    "Filter rows in a DataFrame where score > 80.",
    "Add a new column to a DataFrame that is the product of two other columns.",
    "Sort a DataFrame by age in descending order.",
    "Replace missing values in a DataFrame with the column mean.",
    "Group a DataFrame by 'category' and calculate the mean of 'value'.",
    "Convert a 'date' column to datetime format and extract the month.",
    "Drop duplicate rows from a DataFrame.",
    "Create a pivot table from a DataFrame with sales data.",
    "Merge two DataFrames on a common column.",
    "Plot a bar chart of average scores per category from a DataFrame.",
    "Rename the columns of a DataFrame.",
    "Apply a lambda function to transform a column of text to lowercase.",
    "Create a column that bins numeric scores into letter grades.",
    "Find the correlation matrix between numeric columns in a DataFrame.",
]

matplotlib_problems = [
    "Plot a line chart of y = x^2 for x from -10 to 10.",
    "Display a grayscale image using plt.imshow.",
    "Create a bar chart of counts of unique values in a list.",
    "Plot a histogram of 1000 normally distributed random values.",
    "Create a scatter plot with labeled axes and title.",
    "Use subplots to display 4 plots in a 2x2 grid.",
    "Add a legend to a plot with two different lines.",
    "Plot an image with a bounding box rectangle drawn on it.",
    "Plot multiple lines in the same graph with different styles.",
    "Create a heatmap using imshow on a 10x10 random matrix.",
    "Visualize missing data in a DataFrame using a heatmap (e.g., with seaborn).",
    "Plot a pie chart of percentage distribution of categories.",
    "Draw a filled contour plot of z = sin(x)*cos(y).",
    "Animate a moving sine wave (optional advanced).",
    "Plot RGB image channels separately using subplots.",
]

# Randomly select problems
problems = random.sample(numpy_problems * 7, 40) + \
           random.sample(pandas_problems * 6, 30) + \
           random.sample(matplotlib_problems * 6, 30)

# Shuffle problems
random.shuffle(problems)

# Show as a DataFrame
import pandas as pd
df = pd.DataFrame({"Problem #": range(1, 101), "Task": problems})
print(df.to_string(index=False))
