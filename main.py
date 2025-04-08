import pandas as pd


pickle_file = "static_dataframe_imputed_zscore.pkl"
df = pd.read_pickle(pickle_file)
print(df.dtypes)


