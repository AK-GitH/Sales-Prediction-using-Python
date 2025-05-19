import pandas as pd

# Load dataset
df = pd.read_csv("Advertising.csv")

# Drop unnecessary index column if present
df = df.drop(columns=["Unnamed: 0"], errors='ignore')

# Remove duplicate rows
df = df.drop_duplicates().reset_index(drop=True)

# Define columns
cat_cols = []  # No categorical columns in this dataset
num_cols = [col for col in df.columns if col not in cat_cols + ["Sales"]]

# Handle missing values for numeric columns
for col in num_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# Handle missing values for categorical columns (if any)
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])
        df[col] = df[col].astype("category")

# Remove extreme outliers in Sales
price_threshold = df["Sales"].quantile(0.95)
df = df[df["Sales"] < price_threshold]

# Save cleaned dataset
df.to_csv("cleaned_advertising.csv", index=False)
print("Data has been cleaned & saved as 'cleaned_advertising.csv'")
