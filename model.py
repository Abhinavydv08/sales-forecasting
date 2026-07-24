import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load Dataset
df = pd.read_csv("SuperStore_Sales_Dataset.csv")

# Rename column
df.rename(columns={"Row ID+O6G3A1:R6": "Row ID"}, inplace=True)

# Remove unnecessary columns
for col in ["Returns", "ind1", "ind2"]:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

# Convert date
df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)

# Create Month & Year columns
df["Month"] = df["Order Date"].dt.month
df["Year"] = df["Order Date"].dt.year

# Features
X = df[["Quantity", "Profit", "Month", "Year"]]

# Target
y = df["Sales"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Save Model
joblib.dump(model, "sales_model.pkl")

print("✅ Model Saved Successfully!")