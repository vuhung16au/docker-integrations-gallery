#!/usr/bin/env python3
"""
Data Preparation Script for Heart Disease Prediction
Handles data loading, cleaning, and preprocessing
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_dataset(file_path='Dataset-Heart/heart.csv'):
    """Load the heart disease dataset"""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def clean_dataset(df):
    """Clean the dataset by removing faulty data points"""
    original_shape = df.shape
    
    # Remove faulty data as mentioned in the reference notebook
    # Data #93, 159, 164, 165 and 252 have ca=4 which is incorrect
    # Data #49 and 282 have thal = 0, also incorrect
    df_cleaned = df.copy()
    
    # Remove rows with ca=4 (incorrect values)
    df_cleaned = df_cleaned[df_cleaned['ca'] != 4]
    
    # Remove rows with thal=0 (incorrect values)
    df_cleaned = df_cleaned[df_cleaned['thal'] != 0]
    
    # Reset index
    df_cleaned = df_cleaned.reset_index(drop=True)
    
    logger.info(f"Dataset cleaned. Removed {original_shape[0] - df_cleaned.shape[0]} faulty records")
    logger.info(f"New shape: {df_cleaned.shape}")
    
    return df_cleaned

def rename_columns(df):
    """Rename columns for better readability"""
    column_mapping = {
        'age': 'age',
        'sex': 'sex',
        'cp': 'chest_pain_type',
        'trestbps': 'resting_blood_pressure',
        'chol': 'cholesterol',
        'fbs': 'fasting_blood_sugar',
        'restecg': 'resting_electrocardiogram',
        'thalach': 'max_heart_rate_achieved',
        'exang': 'exercise_induced_angina',
        'oldpeak': 'st_depression',
        'slope': 'st_slope',
        'ca': 'num_major_vessels',
        'thal': 'thalassemia',
        'target': 'target'
    }
    
    df_renamed = df.rename(columns=column_mapping)
    logger.info("Columns renamed for better readability")
    return df_renamed

def decode_categorical_features(df):
    """Decode categorical features to their medical meaning"""
    df_decoded = df.copy()
    
    # Decode sex
    df_decoded['sex'] = df_decoded['sex'].map({0: 'female', 1: 'male'})
    
    # Decode chest pain type
    chest_pain_mapping = {
        0: 'typical angina',
        1: 'atypical angina', 
        2: 'non-anginal pain',
        3: 'asymptomatic'
    }
    df_decoded['chest_pain_type'] = df_decoded['chest_pain_type'].map(chest_pain_mapping)
    
    # Decode fasting blood sugar
    df_decoded['fasting_blood_sugar'] = df_decoded['fasting_blood_sugar'].map({
        0: 'lower than 120mg/ml',
        1: 'greater than 120mg/ml'
    })
    
    # Decode resting electrocardiogram
    ecg_mapping = {
        0: 'normal',
        1: 'ST-T wave abnormality',
        2: 'left ventricular hypertrophy'
    }
    df_decoded['resting_electrocardiogram'] = df_decoded['resting_electrocardiogram'].map(ecg_mapping)
    
    # Decode exercise induced angina
    df_decoded['exercise_induced_angina'] = df_decoded['exercise_induced_angina'].map({
        0: 'no',
        1: 'yes'
    })
    
    # Decode ST slope
    slope_mapping = {
        0: 'upsloping',
        1: 'flat',
        2: 'downsloping'
    }
    df_decoded['st_slope'] = df_decoded['st_slope'].map(slope_mapping)
    
    # Decode thalassemia
    thal_mapping = {
        1: 'fixed defect',
        2: 'normal',
        3: 'reversable defect'
    }
    df_decoded['thalassemia'] = df_decoded['thalassemia'].map(thal_mapping)
    
    logger.info("Categorical features decoded to medical terminology")
    return df_decoded

def prepare_features_for_training(df):
    """Prepare features for model training (convert back to numerical)"""
    # For training, we need numerical values, so we'll use the original encoded values
    # This function prepares the data in the format expected by the models
    
    # Select only the feature columns (exclude target)
    feature_columns = [col for col in df.columns if col != 'target']
    
    # Create feature matrix X and target vector y
    X = df[feature_columns].values
    y = df['target'].values
    
    logger.info(f"Features prepared for training. X shape: {X.shape}, y shape: {y.shape}")
    
    return X, y, feature_columns

def split_data(X, y, test_size=0.25, random_state=42):
    """Split data into training and testing sets"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Data split - Training: {X_train.shape}, Testing: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def main():
    """Main function to run the data preparation pipeline"""
    logger.info("Starting data preparation pipeline...")
    
    try:
        # Load dataset
        df = load_dataset()
        
        # Clean dataset
        df_cleaned = clean_dataset(df)
        
        # Rename columns for readability
        df_renamed = rename_columns(df_cleaned)
        
        # Decode categorical features for EDA
        df_decoded = decode_categorical_features(df_renamed)
        
        # Save processed data for EDA
        os.makedirs('processed_data', exist_ok=True)
        df_decoded.to_csv('processed_data/heart_disease_processed.csv', index=False)
        logger.info("Processed data saved for EDA")
        
        # Prepare features for training (using original encoded values)
        X, y, feature_names = prepare_features_for_training(df_cleaned)
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # Save training data
        np.save('processed_data/X_train.npy', X_train)
        np.save('processed_data/X_test.npy', X_test)
        np.save('processed_data/y_train.npy', y_train)
        np.save('processed_data/y_test.npy', y_test)
        
        # Save feature names
        with open('processed_data/feature_names.txt', 'w') as f:
            for name in feature_names:
                f.write(name + '\n')
        
        logger.info("Data preparation completed successfully!")
        logger.info(f"Training set: {X_train.shape}")
        logger.info(f"Testing set: {X_test.shape}")
        logger.info(f"Feature names: {feature_names}")
        
        return X_train, X_test, y_train, y_test, feature_names
        
    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        raise

if __name__ == "__main__":
    main()
