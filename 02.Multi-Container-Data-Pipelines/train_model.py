#!/usr/bin/env python3
"""
Model Training Script for Multi-Container Data Pipeline
This script reads data from PostgreSQL, trains a linear regression model, and saves it.
"""

import pandas as pd
import numpy as np
import psycopg2
import pickle
import os
import time
from sqlalchemy import create_engine, text
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def wait_for_database():
    """Wait for the database to be ready"""
    max_attempts = 30
    attempt = 0
    
    while attempt < max_attempts:
        try:
            # Try to connect to the database
            conn = psycopg2.connect(
                host=os.getenv('DB_HOST', 'database'),
                database=os.getenv('POSTGRES_DB', 'dsproject'),
                user=os.getenv('POSTGRES_USER', 'postgres'),
                password=os.getenv('POSTGRES_PASSWORD', 'postgres'),
                port=os.getenv('DB_PORT', '5432')
            )
            conn.close()
            print("✅ Database connection successful!")
            return True
        except psycopg2.OperationalError as e:
            attempt += 1
            print(f"⏳ Waiting for database... (attempt {attempt}/{max_attempts})")
            time.sleep(2)
    
    print("❌ Failed to connect to database after maximum attempts")
    return False

def load_data(engine):
    """Load training data from PostgreSQL database"""
    print("📥 Loading training data from database...")
    
    try:
        # Load data from database
        query = "SELECT x, y FROM linear_data ORDER BY x"
        df = pd.read_sql(query, engine)
        
        if df.empty:
            raise ValueError("No data found in database. Please run data preparation first.")
        
        print(f"✅ Loaded {len(df)} records from database")
        print(f"📊 Data shape: {df.shape}")
        print(f"📈 Data range: x=[{df['x'].min():.2f}, {df['x'].max():.2f}], y=[{df['y'].min():.2f}, {df['y'].max():.2f}]")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise

def prepare_features(df):
    """Prepare features for training"""
    print("🔧 Preparing features for training...")
    
    # Prepare X and y
    X = df['x'].values.reshape(-1, 1)
    y = df['y'].values
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"✅ Features prepared:")
    print(f"   - Training set: {X_train.shape[0]} samples")
    print(f"   - Testing set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train the linear regression model"""
    print("🤖 Training linear regression model...")
    
    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print("✅ Model training completed!")
    print(f"📊 Model coefficients: {model.coef_[0]:.4f}")
    print(f"📊 Model intercept: {model.intercept_:.4f}")
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model"""
    print("📊 Evaluating model performance...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("✅ Model evaluation completed!")
    print(f"📈 Mean Squared Error: {mse:.4f}")
    print(f"📈 Root Mean Squared Error: {rmse:.4f}")
    print(f"📈 R² Score: {r2:.4f}")
    
    # Show some predictions
    print("\n📋 Sample predictions:")
    for i in range(min(5, len(X_test))):
        actual = y_test[i]
        predicted = y_pred[i]
        x_val = X_test[i][0]
        print(f"   x={x_val:.2f}: actual={actual:.2f}, predicted={predicted:.2f}")
    
    return y_pred

def save_model(model, model_path='model.pkl'):
    """Save the trained model"""
    print(f"💾 Saving model to {model_path}...")
    
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"✅ Model saved successfully to {model_path}")
        
        # Verify the saved model
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        
        # Test prediction
        test_x = np.array([[5.0]])
        test_pred = loaded_model.predict(test_x)[0]
        expected = 2 * 5.0 + 1
        print(f"🧪 Model verification: x=5 → predicted={test_pred:.4f} (expected: {expected:.4f})")
        
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        raise

def main():
    """Main function to train the model"""
    print("🚀 Starting model training pipeline...")
    
    # Wait for database to be ready
    if not wait_for_database():
        return
    
    # Database connection parameters
    db_url = f"postgresql://{os.getenv('POSTGRES_USER', 'postgres')}:{os.getenv('POSTGRES_PASSWORD', 'postgres')}@{os.getenv('DB_HOST', 'database')}:{os.getenv('DB_PORT', '5432')}/{os.getenv('POSTGRES_DB', 'dsproject')}"
    
    try:
        # Create database engine
        engine = create_engine(db_url)
        
        # Load data
        df = load_data(engine)
        
        # Prepare features
        X_train, X_test, y_train, y_test = prepare_features(df)
        
        # Train model
        model = train_model(X_train, y_train)
        
        # Evaluate model
        y_pred = evaluate_model(model, X_test, y_test)
        
        # Save model
        save_model(model)
        
        print("🎉 Model training pipeline completed successfully!")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        raise

if __name__ == '__main__':
    main()
