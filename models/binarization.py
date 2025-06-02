# binarization.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def thermometer_encode(data, num_bins, feature_ranges):
    """
    Applies thermometer encoding to numerical data.

    Args:
        data (np.ndarray): Numerical input data (features).
                           Shape: (n_samples, n_features).
        num_bins (int): The number of bins (bits) to use for encoding each feature.
        feature_ranges (dict): A dictionary where keys are feature indices and values are
                               tuples (min_val, max_val) for that feature's range.
                               Example: {0: (0.0, 1.0), 1: (0.0, 1.0), ...}

    Returns:
        np.ndarray: Binarized data. Shape: (n_samples, n_features * num_bins).
                    Each row is a concatenated binary pattern.
    """
    n_samples, n_features = data.shape
    binarized_data = np.zeros((n_samples, n_features * num_bins), dtype=int)

    for i in range(n_features):
        min_val, max_val = feature_ranges[i]
        
        # Handle cases where min_val == max_val to avoid division by zero
        if max_val - min_val == 0:
            # If feature is constant, encode it as all zeros or all ones based on value
            # For simplicity, we'll encode it as all zeros in this case.
            # A more robust approach might remove constant features earlier.
            pass 
        else:
            # Calculate bin width
            bin_width = (max_val - min_val) / num_bins
            
            for j in range(n_samples):
                val = data[j, i]
                
                # Determine which bin the value falls into
                # Clip value to ensure it's within the defined range
                clipped_val = np.clip(val, min_val, max_val)
                bin_idx = int(np.floor((clipped_val - min_val) / bin_width))
                
                # Ensure bin_idx does not exceed num_bins - 1 for max_val
                if bin_idx == num_bins:
                    bin_idx = num_bins - 1
                
                # Set bits for thermometer encoding
                start_idx = i * num_bins
                binarized_data[j, start_idx : start_idx + bin_idx + 1] = 1
                
    return binarized_data

if __name__ == "__main__":
    print("--- Testing binarization.py ---")
    
    # Create dummy numerical data
    np.random.seed(42)
    X_dummy = np.random.rand(5, 3) * 10 # 5 samples, 3 features, values 0-10
    
    # Simulate feature ranges (usually from MinMaxScaler on training data)
    scaler = MinMaxScaler()
    scaler.fit(X_dummy)
    feature_ranges_dummy = {i: (scaler.data_min_[i], scaler.data_max_[i]) for i in range(X_dummy.shape[1])}
    
    num_bins = 4
    
    print("\nOriginal Dummy Data:")
    print(X_dummy)
    print("\nFeature Ranges:")
    print(feature_ranges_dummy)
    
    # Binarize the dummy data
    X_bin_dummy = thermometer_encode(X_dummy, num_bins, feature_ranges_dummy)
    
    print(f"\nBinarized Data (shape: {X_bin_dummy.shape}):")
    print(X_bin_dummy)
    
    # Verify the structure (e.g., first feature's encoding for first sample)
    print("\nVerifying first feature's encoding (first sample):")
    feature_val = X_dummy[0, 0]
    min_val, max_val = feature_ranges_dummy[0]
    bin_width = (max_val - min_val) / num_bins
    bin_idx = int(np.floor((np.clip(feature_val, min_val, max_val) - min_val) / bin_width))
    if bin_idx == num_bins: bin_idx = num_bins - 1
    
    expected_bin_pattern = np.zeros(num_bins, dtype=int)
    expected_bin_pattern[:bin_idx + 1] = 1
    
    print(f"Original value for feature 0, sample 0: {feature_val:.2f}")
    print(f"Expected bin index: {bin_idx}")
    print(f"Expected binary pattern: {expected_bin_pattern}")
    print(f"Actual binary pattern from function: {X_bin_dummy[0, :num_bins]}")
