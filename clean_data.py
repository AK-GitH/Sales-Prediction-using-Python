import pandas as pd

# Load dataset
df = pd.read_csv("Advertising.csv")

# Drop unnecessary index column if present
df = df.drop(columns=["Unnamed: 0"], errors='ignore')

# Remove duplicate rows
df = df.drop_duplicates().reset_index(drop=True)

# Define columns
num_cols = [col for col in df.columns if col != "Sales"]

# Handle missing values for numeric columns
for col in num_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# Remove extreme outliers in Sales using IQR
Q1 = df["Sales"].quantile(0.25)
Q3 = df["Sales"].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR
df = df[df["Sales"] < upper_bound]

# Save cleaned dataset
df.to_csv("cleaned_advertising.csv", index=False)
print("Data has been cleaned & saved as 'cleaned_advertising.csv'")
