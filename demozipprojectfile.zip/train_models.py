# train_models.py

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression # Added LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor # Added RandomForestRegressor
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score # Added regression metrics
from data_preprocessing import preprocess_data

def train_and_evaluate_classifier(X, y, disease_name):
    """
    Splits data, trains classification models, and saves the best one.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=500),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='linear', random_state=42)
    }

    best_model_name = ''
    best_model = None
    best_accuracy = 0

    print(f"\n--- Training Classifier Models for {disease_name} ---")
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Accuracy for {name}: {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
            best_model = model

    print(f"\nBest model for {disease_name}: {best_model_name} with Accuracy: {best_accuracy:.4f}")
    
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    model_path = os.path.join(models_dir, f'{disease_name}_model.pkl')
    joblib.dump(best_model, model_path)
    print(f"Best model saved as '{model_path}'")
    
    return best_model

def train_and_evaluate_regressor(X, y, disease_name):
    """
    Splits data, trains regression models, and saves the best one.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models_reg = {
        'Linear Regression': LinearRegression(),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    best_model_name = ''
    best_model = None
    best_score = float('inf')  # Using Mean Squared Error, lower is better
    
    print(f"\n--- Training Regressor Models for {disease_name} ---")
    
    for name, model in models_reg.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R-squared: {r2:.4f}")
        
        if mse < best_score:
            best_score = mse
            best_model_name = name
            best_model = model

    print(f"\nBest model for {disease_name}: {best_model_name} with MSE: {best_score:.4f}")
    
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    model_path = os.path.join(models_dir, f'{disease_name}_model.pkl')
    joblib.dump(best_model, model_path)
    print(f"Best model saved as '{model_path}'")
    
    return best_model


if __name__ == '__main__':
    # --- Train Diabetes Model ---
    diabetes_df, diabetes_scaler = preprocess_data(os.path.join('data', 'diabetes.csv'), 'diabetes')
    if diabetes_df is not None:
        X_diabetes = diabetes_df.drop('Outcome', axis=1)
        y_diabetes = diabetes_df['Outcome']
        joblib.dump(diabetes_scaler, os.path.join('models', 'diabetes_scaler.pkl'))
        train_and_evaluate_classifier(X_diabetes, y_diabetes, 'diabetes')

    # --- Train Heart Disease Model ---
    heart_df, heart_scaler = preprocess_data(os.path.join('data', 'heart_disease.csv'), 'heart')
    if heart_df is not None:
        X_heart = heart_df.drop('target', axis=1)
        y_heart = heart_df['target']
        joblib.dump(heart_scaler, os.path.join('models', 'heart_scaler.pkl'))
        train_and_evaluate_classifier(X_heart, y_heart, 'heart')

    # --- Train Parkinson's Model as Regressor ---
    parkinsons_df, parkinsons_scaler = preprocess_data(os.path.join('data', 'parkinsons.csv'), 'parkinsons')
    if parkinsons_df is not None:
        X_parkinsons = parkinsons_df.drop('status', axis=1)
        y_parkinsons = parkinsons_df['status']
        joblib.dump(parkinsons_scaler, os.path.join('models', 'parkinsons_scaler.pkl'))
        # Call the regressor function for Parkinson's
        train_and_evaluate_regressor(X_parkinsons, y_parkinsons, 'parkinsons')