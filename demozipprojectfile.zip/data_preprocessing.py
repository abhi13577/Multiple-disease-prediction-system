# data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def preprocess_data(file_path, disease_name):
    """
    Loads, cleans, and scales a disease dataset.
    Returns the preprocessed DataFrame and the fitted scaler.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Make sure the file is in the 'data' folder.")
        return None, None

    # --- Data Cleaning (handling 0s in Diabetes dataset) ---
    if disease_name == 'diabetes':
        cols_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in cols_to_impute:
            df[col] = df[col].replace(0, df[col][df[col] != 0].median())
    
    # --- Corrected Part: Drop non-numeric columns before scaling ---
    if disease_name == 'parkinsons':
        # Drop the 'name' column as it is a string and not a feature
        if 'name' in df.columns:
            df = df.drop('name', axis=1)

    # --- Feature Scaling ---
    scaler = StandardScaler()
    
    # Exclude the target variable from scaling
    if disease_name == 'diabetes':
        target = 'Outcome'
    elif disease_name == 'heart':
        target = 'target'
    elif disease_name == 'parkinsons':
        target = 'status'
    else:
        return None, None
        
    if target not in df.columns:
        print(f"Error: Target column '{target}' not found in the {disease_name} dataset.")
        return None, None

    # Separate features and target
    X = df.drop(target, axis=1)
    y = df[target]
    
    # Fit and transform the features using the scaler
    # This line now receives a DataFrame with only numerical values
    X_scaled = scaler.fit_transform(X)
    
    # Re-assemble the DataFrame with scaled features
    scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    scaled_df[target] = y

    print(f"Data for {disease_name} preprocessed and scaled successfully.")
    
    return scaled_df, scaler

# ... (rest of the file remains the same)

if __name__ == '__main__':
    print("Running data_preprocessing.py in standalone mode.")
    diabetes_df, diabetes_scaler = preprocess_data(os.path.join('data', 'diabetes.csv'), 'diabetes')
    if diabetes_df is not None:
        print("\nDiabetes DataFrame description after preprocessing and scaling:")
        print(diabetes_df.describe())