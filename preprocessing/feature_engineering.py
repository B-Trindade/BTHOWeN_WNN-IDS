# feature_engineering.py

from select_dataset import select_dataset

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE # Recursive Feature Elimination

def encode_categorical_features(df, target_column='Label'):
    """
    Encodes categorical features using Label Encoding, excluding the target column.
    Handles columns with mixed types by converting to string before encoding.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        target_column (str): The name of the target column (will not be encoded).

    Returns:
        pandas.DataFrame: DataFrame with categorical features encoded.
        dict: A dictionary mapping original column names to their LabelEncoder instances.
    """
    df_encoded = df.copy()
    encoders = {}
    
    for col in df_encoded.columns:
        if col != target_column and df_encoded[col].dtype == 'object':
            # Convert to string to handle potential mixed types gracefully
            df_encoded[col] = df_encoded[col].astype(str)
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            encoders[col] = le
            print(f"Encoded categorical column: {col}")
    
    # Handle the target column separately if it's categorical
    if target_column in df_encoded.columns and df_encoded[target_column].dtype == 'object':
        df_encoded[target_column] = df_encoded[target_column].astype(str)
        le_target = LabelEncoder()
        df_encoded[target_column] = le_target.fit_transform(df_encoded[target_column])
        encoders[target_column] = le_target
        print(f"Encoded target column: {target_column}")

    return df_encoded, encoders

def scale_numerical_features(df, target_column='Label'):
    """
    Scales numerical features using StandardScaler, excluding the target column.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        target_column (str): The name of the target column (will not be scaled).

    Returns:
        pandas.DataFrame: DataFrame with numerical features scaled.
        sklearn.preprocessing.StandardScaler: The scaler fitted on the data.
    """
    df_scaled = df.copy()
    numerical_cols = df_scaled.select_dtypes(include=np.number).columns.tolist()
    if target_column in numerical_cols:
        numerical_cols.remove(target_column)

    if not numerical_cols:
        print("No numerical columns to scale (excluding target).")
        return df_scaled, None

    scaler = StandardScaler()
    df_scaled[numerical_cols] = scaler.fit_transform(df_scaled[numerical_cols])
    print(f"Scaled numerical columns: {numerical_cols}")
    return df_scaled, scaler

def perform_correlation_analysis(df, target_column='Label', plot=True):
    """
    Performs correlation analysis on numerical features and visualizes the correlation matrix.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        target_column (str): The name of the target column.
        plot (bool): Whether to plot the heatmap of the correlation matrix.

    Returns:
        pandas.DataFrame: The correlation matrix.
    """
    # Ensure numerical features are used for correlation
    numerical_df = df.select_dtypes(include=np.number)
    
    if numerical_df.empty:
        print("No numerical columns available for correlation analysis.")
        return pd.DataFrame()

    correlation_matrix = numerical_df.corr()
    print("\nCorrelation Matrix (first 5x5 block):")
    print(correlation_matrix.head(5).iloc[:, :5])

    if plot:
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Correlation Matrix of Features')
        plt.show()
    
    if target_column in correlation_matrix.columns:
        print(f"\nCorrelation with target column '{target_column}':")
        print(correlation_matrix[target_column].sort_values(ascending=False))

    return correlation_matrix

def select_features_univariate(X, y, k=10, score_func=f_classif):
    """
    Selects the top K features based on univariate statistical tests.

    Args:
        X (pandas.DataFrame): Features DataFrame.
        y (pandas.Series): Target Series.
        k (int): Number of top features to select.
        score_func (callable): Scoring function (e.g., f_classif, mutual_info_classif).

    Returns:
        list: Names of the selected features.
    """
    print(f"\n--- Univariate Feature Selection using {score_func.__name__} (Top {k} features) ---")
    selector = SelectKBest(score_func=score_func, k=k)
    selector.fit(X, y)
    selected_features_mask = selector.get_support()
    selected_features = X.columns[selected_features_mask].tolist()
    print(f"Selected features: {selected_features}")
    
    scores = pd.DataFrame({'Feature': X.columns, 'Score': selector.scores_, 'P-value': selector.pvalues_})
    scores = scores.sort_values(by='Score', ascending=False).reset_index(drop=True)
    print("\nFeature Scores:")
    print(scores.head(k))
    
    return selected_features

def select_features_rfe(X, y, estimator=None, n_features_to_select=10):
    """
    Selects features using Recursive Feature Elimination (RFE).

    Args:
        X (pandas.DataFrame): Features DataFrame.
        y (pandas.Series): Target Series.
        estimator: The base estimator to use for feature ranking (e.g., RandomForestClassifier).
                   If None, RandomForestClassifier is used by default.
        n_features_to_select (int): The number of features to select.

    Returns:
        list: Names of the selected features.
    """
    print(f"\n--- Recursive Feature Elimination (RFE) (Top {n_features_to_select} features) ---")
    if estimator is None:
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        print("Using RandomForestClassifier as default estimator for RFE.")

    selector = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=1)
    selector.fit(X, y)
    selected_features_mask = selector.get_support()
    selected_features = X.columns[selected_features_mask].tolist()
    print(f"Selected features: {selected_features}")
    
    ranking = pd.DataFrame({'Feature': X.columns, 'Ranking': selector.ranking_})
    ranking = ranking.sort_values(by='Ranking').reset_index(drop=True)
    print("\nFeature Rankings (lower is better):")
    print(ranking.head(n_features_to_select))

    return selected_features


if __name__ == "__main__":
    # Example Usage:
    
    df = select_dataset(attack_filter='DDoS', data_path='./data/Portscan-DDos-Botnet-Friday.parquet')
    
    print("Original Dummy DataFrame shape:", df.shape)
    print("\nOriginal Dummy DataFrame Info:")
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

    # Step 3: Perform Correlation Analysis
    print("\n--- Performing Correlation Analysis ---")
    correlation_matrix = perform_correlation_analysis(df_scaled.copy(), target_column='Label', plot=True)

    # Step 4: Feature Selection using SelectKBest (ANOVA F-value)
    selected_features_f_classif = select_features_univariate(X, y, k=5, score_func=f_classif)
    print(f"\nFeatures selected by F-classif: {selected_features_f_classif}")

    # Step 5: Feature Selection using SelectKBest (Mutual Information)
    selected_features_mutual_info = select_features_univariate(X, y, k=5, score_func=mutual_info_classif)
    print(f"\nFeatures selected by Mutual Information: {selected_features_mutual_info}")

    # Step 6: Feature Selection using Recursive Feature Elimination (RFE)
    # Note: RFE requires a classifier that provides feature importances (e.g., RandomForestClassifier)
    selected_features_rfe = select_features_rfe(X, y, n_features_to_select=5)
    print(f"\nFeatures selected by RFE: {selected_features_rfe}")
