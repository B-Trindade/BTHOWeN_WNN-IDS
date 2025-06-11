# wisardpkg_implementation.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import wisardpkg as wp # Import the wisardpkg module

# Import the binarization function from the new file
from helpers.binarization import thermometer_encode
# Import the select_data function from the new select_features.py file
from preprocessing.select_features import select_data

if __name__ == "__main__":
    print("--- WiSARDpkg Model Implementation ---")

    # 1. Obtain Preprocessed and Feature-Selected Data
    #    The select_data() function is expected to return X (features) and y (labels)
    #    that have already gone through:
    #    - Merging of .parquet files
    #    - Data cleaning (duplicates, missing values, constant columns handled)
    #    - Feature selection (top 10 features chosen)
    #    - Filtering for 'DDoS' or 'Benign' labels (mapped to integers)
    
    X, y = select_data()

    # Determine number of features from the obtained X
    num_features = X.shape[1]
    # Determine unique labels from the obtained y
    unique_labels = np.unique(y)
    
    print(f"\nData obtained from select_features.py:")
    print(f"Features (X) shape: {X.shape}")
    print(f"Labels (y) shape: {y.shape}")
    print(f"Unique labels in y: {unique_labels}")
    print(f"Number of features: {num_features}")

    # 2. Prepare Data for WiSARDpkg
    #    - Split data into training and testing sets
    #    - Determine feature ranges for thermometer encoding (important to do on training data only)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y # Stratify to maintain label ratio
    )
    
    print(f"\nTrain set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    print(f"Train label distribution: {pd.Series(y_train).value_counts(normalize=True)}")
    print(f"Test label distribution: {pd.Series(y_test).value_counts(normalize=True)}")

    # Determine feature ranges from the TRAINING data for thermometer encoding
    # Use MinMaxScaler to get min/max for each feature. This is crucial for consistent binarization.
    scaler = MinMaxScaler()
    scaler.fit(X_train) 
    
    feature_ranges = {i: (scaler.data_min_[i], scaler.data_max_[i]) for i in range(num_features)}
    print(f"\nFeature ranges (from training data): {feature_ranges}")

    # Apply thermometer encoding to both training and test sets using the imported function
    # num_bins_per_feature is a hyperparameter for binarization
    num_bins_per_feature = 8 
    
    X_train_bin = thermometer_encode(X_train, num_bins_per_feature, feature_ranges)
    X_test_bin = thermometer_encode(X_test, num_bins_per_feature, feature_ranges)

    print(f"\nBinarized training data shape: {X_train_bin.shape}")
    print(f"Binarized test data shape: {X_test_bin.shape}")
    print(f"Example binarized pattern (first row of X_train_bin):\n{X_train_bin[0]}")

    # Calculate total binary inputs for WiSARDpkg
    total_binary_inputs = num_features * num_bins_per_feature
    print(f"Total binary inputs for WiSARDpkg: {total_binary_inputs}")

    # 3. Initialize and Train WiSARDpkg Model
    # wisardpkg expects binary inputs (0s and 1s) and integer labels.
    # The 'tuple_size' parameter in wisardpkg is equivalent to 'n_tuple' in our custom model.
    
    tuple_size = 4 # Example tuple size (hyperparameter)
    
    # wisardpkg expects lists of lists for input patterns and a list for labels
    X_train_bin_list = X_train_bin.tolist()
    y_train_list = y_train.tolist()

    print(f"\nInitializing WiSARDpkg model with tuple_size={tuple_size}...")
    wisard = wp.Wisard(
        tuple_size=tuple_size,
        bleaching=1, # Default bleaching for basic tie-breaking (no complex iterative reduction)
        verbose=True
    )
    
    print("\nStarting WiSARDpkg training...")
    wisard.train(X_train_bin_list, y_train_list)
    print("WiSARDpkg training complete.")

    # 4. Make Predictions and Evaluate
    # wisardpkg.predict also expects a list of lists for input patterns
    X_test_bin_list = X_test_bin.tolist()

    print("\nStarting WiSARDpkg prediction...")
    y_pred = wisard.predict(X_test_bin_list)
    print("WiSARDpkg prediction complete.")

    print("\n--- Model Evaluation (using wisardpkg) ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\n--- Experimentation Notes ---")
    print("The 'wisardpkg' module provides a highly optimized C++ implementation.")
    print("Its 'bleaching' parameter works differently than our custom model's iterative tie-breaking.")
    print("A 'bleaching' value of 1 in wisardpkg means no explicit bleaching (random tie-break if activations are equal).")
    print("Values > 1 imply a threshold for activation difference to break ties.")
    print("For advanced bleaching strategies, you might need to explore wisardpkg's documentation or implement post-processing.")
