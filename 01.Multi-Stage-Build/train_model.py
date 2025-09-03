#!/usr/bin/env python3
"""
Simple Linear Regression Model Training Script
This script trains a linear regression model and saves it for the serving stage.
"""

import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def train_model():
    """Train a simple linear regression model"""
    print("Starting model training...")
    
    # Generate synthetic data: y = 2x + 1 + noise
    np.random.seed(42)
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = 2 * X.flatten() + 1 + np.random.normal(0, 0.5, 100)
    
    # Create and train the model
    model = LinearRegression()
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f"Training completed!")
    print(f"Model coefficients: {model.coef_[0]:.4f}")
    print(f"Model intercept: {model.intercept_:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Save the trained model
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Model saved as 'model.pkl'")
    
    # Verify the model works
    test_X = np.array([[5.0]])
    test_pred = model.predict(test_X)[0]
    expected = 2 * 5.0 + 1
    print(f"Test prediction for x=5: {test_pred:.4f} (expected: {expected:.4f})")

if __name__ == '__main__':
    train_model()
