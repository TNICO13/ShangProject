import pandas as pd
import numpy as np

def standardize_floats(df):
    """
    Convert all float columns in the DataFrame to z-scores.
    If a column's standard deviation is zero (or very close to zero),
    that column is left unchanged (or set to zero) to avoid division by zero.
    """
    float_cols = df.select_dtypes(include=['float64']).columns
    for col in float_cols:
        col_mean = df[col].mean()
        col_std = df[col].std()
        if np.isclose(col_std, 0):
            print(f"Column '{col}' has near-zero standard deviation; leaving as is.")
            continue
        df[col] = (df[col] - col_mean) / col_std
    return df

# ---------------------------
# Load the non-imputed DataFrame.
# ---------------------------
df = pd.read_pickle("static_dataframe.pkl")
# Standardize float columns in the non-imputed DataFrame.
df_standardized = standardize_floats(df.copy())

# Optionally, save the standardized non-imputed DataFrame.
df_standardized.to_pickle("static_dataframe_zscore.pkl")

# ---------------------------
# Load the imputed DataFrame.
# ---------------------------
df_imputed = pd.read_pickle("static_dataframe_imputed.pkl")
# Standardize float columns in the imputed DataFrame.
df_imputed_standardized = standardize_floats(df_imputed.copy())

# Optionally, save the standardized imputed DataFrame.
df_imputed_standardized.to_pickle("static_dataframe_imputed_zscore.pkl")

# Display the first few rows of each for verification.
print("Non-imputed standardized data (first 5 rows):")
print(df_standardized.head())

print("\nImputed standardized data (first 5 rows):")
print(df_imputed_standardized.head())
