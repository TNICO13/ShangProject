import pandas as pd

# 1. Load the DataFrame from the pickle file
pickle_file = "static_dataframe.pkl"
df = pd.read_pickle(pickle_file)

# Show the count of missing values before imputation
print("Missing values before imputation:")
print(df.isnull().sum())

# 2. Impute Numeric Columns with the Median

# Identify numeric columns (int64 and float64)
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Fill missing numeric values with the median of each column
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# 3. Impute Categorical Columns with the Mode

# Identify categorical columns (object type)
categorical_cols = df.select_dtypes(include=['object']).columns

# Fill missing categorical values with the mode (most frequent value)
for col in categorical_cols:
    if not df[col].mode().empty:
        mode_value = df[col].mode()[0]
        df[col] = df[col].fillna(mode_value)

# Verify that missing values are handled
print("\nMissing values after imputation:")
print(df.isnull().sum())

# 4. Save the updated DataFrame to a new pickle file
imputed_pickle_file = "static_dataframe_imputed.pkl"
df.to_pickle(imputed_pickle_file)
print(f"\nImputed DataFrame saved to {imputed_pickle_file}")