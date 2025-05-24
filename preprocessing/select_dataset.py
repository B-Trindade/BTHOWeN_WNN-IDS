import pandas as pd

def select_dataset(attack_filter:str = 'DDoS', data_path:str = './data/Portscan-DDos-Botnet-Friday.parquet'):
    """
    Selects a dataset from the cleaned CIC-IDS2017 list of datasets and applies preprocessing steps.
    Args:
        attack_filter (str): The type of attack the dataset will keep. Can either be 'Portscan', 'DDoS' or None.
        If None, both 'Portscan' and 'DDoS' will be kept, but 'Botnet' will still be removed.
        data_path (str): Path to the dataset file.
    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """

    # Load the dataset
    df = pd.read_parquet(data_path)

    if df.empty:
        print('The dataset is empty. Please check if file exists and is not corrupted.')
        return pd.DataFrame()
    
    # Filter the dataset based on the attack type specified
    if attack_filter not in ['Portscan', 'DDoS', None]:
        print(f"Invalid attack type '{attack_filter}'. Please choose from 'Portscan', 'DDoS', or 'All'.")
        return pd.DataFrame()
    elif attack_filter:
        # Keep only the specified attack type
        df = df[df['Label'].str.contains(f'Benign|{attack_filter}')]
    # Keep both Portscan and DDoS, but remove Botnet
    df.drop(df[df['Label'].str.contains('Attempted|Botnet')].index, inplace=True)
    return df

if __name__ == "__main__":
    # Example usage
    atk_filter = 'DDoS' #! Change this to 'Portscan' or None 
    df = select_dataset(attack_filter=atk_filter, data_path='./data/Portscan-DDos-Botnet-Friday.parquet')
    if not df.empty:
        print(f"Selected dataset with shape: {df.shape}.")
        print(f"Selected dataset attack type(s): {df['Label'].unique()}.")
        print("="*50)
        print("Data balance: absolute values | normalized")

        counts = df.Label.value_counts(normalize=False)
        norm_counts = df.Label.value_counts(normalize=True)

        print(f"Benign: {counts['Benign']} | {norm_counts['Benign']:.2%}")
        if atk_filter == 'DDoS' or atk_filter is None:
            print(f"DDoS: {counts['DDoS']} | {norm_counts['DDoS']:.2%}")
        if atk_filter == 'Portscan' or atk_filter is None:
            print(f"Portscan: {counts['Portscan']} | {norm_counts['Portscan']:.2%}")
        print(counts)
    else:
        print("Error: No data to display. Verify args and dataset health.")

    