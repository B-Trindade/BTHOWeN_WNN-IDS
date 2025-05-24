# merge_data.py

import pandas as pd
import os

def merge_parquet_files(data_dir='../data'):
    """
    Merges all .parquet files from a specified directory into a single Pandas DataFrame.

    Args:
        data_dir (str): The directory containing the .parquet files.

    Returns:
        pandas.DataFrame: A single DataFrame containing all merged data.
                          Returns an empty DataFrame if no .parquet files are found.
    """
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.parquet')]
    
    if not all_files:
        print(f"No .parquet files found in directory: {data_dir}")
        return pd.DataFrame()

    print(f"Found {len(all_files)} .parquet files in '{data_dir}'. Merging...")
    
    # Initialize an empty list to store DataFrames
    df_list = []
    
    for f_path in all_files:
        try:
            df = pd.read_parquet(f_path)
            df_list.append(df)
            print(f"Successfully loaded {os.path.basename(f_path)}")
        except Exception as e:
            print(f"Error loading {os.path.basename(f_path)}: {e}")
            continue
            
    if not df_list:
        print("No DataFrames were successfully loaded.")
        return pd.DataFrame()

    merged_df = pd.concat(df_list, ignore_index=True)
    print(f"Successfully merged {len(df_list)} files into a single DataFrame with shape: {merged_df.shape}")
    return merged_df

if __name__ == "__main__":
    # Example usage:
    # Ensure you have a 'data' directory in the same location as this script
    # and it contains your .parquet files (e.g., day1.parquet, day2.parquet, etc.)
    
    # Create a dummy 'data' directory and some dummy .parquet files for testing
    # In a real scenario, these files would already exist from the CIC-IDS2017 dataset.
    if not os.path.exists('data'):
        os.makedirs('data')
        print("Created 'data' directory for testing.")
        
        # Create dummy dataframes and save them as parquet
        for i in range(1, 6):
            dummy_df = pd.DataFrame({
                'feature_A': range(i * 10, i * 10 + 5),
                'feature_B': [f'cat_{i}' for _ in range(5)],
                'Label': ['BENIGN'] * 3 + ['ATTACK'] * 2
            })
            dummy_df.to_parquet(f'data/day{i}.parquet', index=False)
            print(f"Created dummy file: data/day{i}.parquet")

    # Merge the files
    full_df = merge_parquet_files()
    
    if not full_df.empty:
        print("\nFirst 5 rows of the merged DataFrame:")
        print(full_df.head())
        print("\nMerged DataFrame Info:")
        full_df.info()
        
        # You might want to save the merged DataFrame for subsequent steps
        # full_df.to_parquet('data/merged_cicids2017.parquet', index=False)
        # print("\nMerged DataFrame saved to 'data/merged_cicids2017.parquet'")
