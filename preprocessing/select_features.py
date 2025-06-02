import pandas as pd
import numpy as np

from preprocessing.feature_engineering import get_fscore_classif, select_dataset, encode_categorical_features, scale_numerical_features

def select_data(dataset:str = "DDoS", path:str = "./data/Portscan-DDos-Botnet-Friday.parquet"):
    """
    TODO
    Args:
        dataset:
        path:
    Returns:
        X: feature values for each observation
        y: list of labels for each observation
    
    """

    df = select_dataset(attack_filter=dataset, data_path=path)

    print("DataFrame shape:", df.shape)
    print("\nDataFrame Info:")
    df.info()

    # Step 1: Encode categorical features
    df_encoded, _ = encode_categorical_features(df.copy(), target_column='Label')
    
    # Step 2: Scale numerical features
    df_scaled, _ = scale_numerical_features(df_encoded.copy(), target_column='Label')

    # Separate features (X) and target (y)
    X = df_scaled.drop(columns=['Label'])
    y = df_scaled['Label']
    
    print("\nDataFrame after Encoding and Scaling (first 5 rows):")
    print(X.head())
    print("\nTarget variable (first 5 values):")
    print(y.head())

    top_features = get_fscore_classif(X=X, y=y, initial_k=60, step=5, top_n=30, plot=False)
    #! Selects the top 10 features, FIXME for variable top k
    X = X[top_features[:10]] 

    return X, y