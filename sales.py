print("Program Started")
import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("SuperStore_Sales_Dataset.csv")

# Data Cleaning
df.rename(columns={"Row ID+O6G3A1:R6": "Row ID"}, inplace=True)
df.drop(columns=["Returns", "ind1", "ind2"], inplace=True)
df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)

# Monthly Sales
monthly_sales = df.groupby(df["Order Date"].dt.to_period("M"))["Sales"].sum()

# Convert Period to Timestamp
monthly_sales.index = monthly_sales.index.to_timestamp()

# Plot
plt.figure(figsize=(12,6))
plt.plot(monthly_sales.index, monthly_sales.values, marker='o')

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.grid(True)

plt.show()


category_sales = df.groupby("Category")["Sales"].sum()

plt.figure(figsize=(8,5))
plt.bar(category_sales.index, category_sales.values)

plt.title("Sales by Category")
plt.xlabel("Category")
plt.ylabel("Sales")

plt.show()


# Sales by Category

category_sales = df.groupby("Category")["Sales"].sum()

plt.figure(figsize=(8,5))

plt.bar(category_sales.index, category_sales.values)

plt.title("Sales by Category")
plt.xlabel("Category")
plt.ylabel("Total Sales")

plt.show()

# Sales by Region

region_sales = df.groupby("Region")["Sales"].sum()

plt.figure(figsize=(8,5))

plt.bar(region_sales.index, region_sales.values)

plt.title("Sales by Region")
plt.xlabel("Region")
plt.ylabel("Total Sales")

plt.show()
print("Program Finished")