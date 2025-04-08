import pandas as pd

# Define the file path
file_path = "data/datafile.csv"

try:
    # Load the CSV file using parameters to handle irregular lines.
    df = pd.read_csv(
        file_path,
        comment='#',  # Skip any line starting with '#'
        on_bad_lines='skip',  # Skip lines with tokenization issues
        engine='python'  # Use the Python engine for more flexible parsing
    )

    num_rows = len(df)
    print(f"The DataFrame has {num_rows} rows.")


    # Save the DataFrame to a pickle file
    pickle_file = "static_dataframe.pkl"
    df.to_pickle(pickle_file)
    print(f"DataFrame saved to {pickle_file}")

    # Later, load the DataFrame from the pickle file
    df_loaded = pd.read_pickle(pickle_file)
    print("DataFrame loaded from pickle:")
    print(df_loaded.head())

    pickle_file = "static_dataframe.pkl"
    df = pd.read_pickle(pickle_file)


    # Assuming df is your DataFrame that's already been loaded.
    # Get the column label at index 1.
    column_label = df.columns[0]

    # Drop duplicate rows based on that column.
    df = df.drop_duplicates(subset=[column_label])
    print(df.columns)
    df.to_pickle(pickle_file)
    # Display the result.

    num_rows = len(df)
    print(f"The DataFrame has {num_rows} rows.")



except Exception as e:
    print("Error reading CSV:", e)
