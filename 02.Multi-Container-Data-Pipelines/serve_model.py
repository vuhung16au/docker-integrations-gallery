#!/usr/bin/env python3
"""
Model Serving Script for Multi-Container Data Pipeline
This script serves the trained linear regression model via Flask API.
"""

from flask import Flask, request, jsonify
import numpy as np
import pickle
import os
import time
import psycopg2
from sqlalchemy import create_engine, text

app = Flask(__name__)

# Global variable for the model
model = None

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

def load_model():
    """Load the trained model"""
    global model
    
    print("📥 Loading trained model...")
    
    try:
        # Try to load the model from file
        if os.path.exists('model.pkl'):
            with open('model.pkl', 'rb') as f:
                model = pickle.load(f)
            print("✅ Model loaded from file successfully!")
        else:
            print("⚠️  Model file not found. Using fallback model.")
            # Fallback to simple model if file not found
            model = SimpleLinearModel()
        
        # Test the model
        test_x = np.array([[5.0]])
        test_pred = model.predict(test_x)[0]
        print(f"🧪 Model test: x=5 → predicted={test_pred:.4f}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("⚠️  Using fallback model.")
        model = SimpleLinearModel()

class SimpleLinearModel:
    """Fallback simple linear model"""
    def __init__(self):
        self.coef_ = 2.0
        self.intercept_ = 1.0
    
    def predict(self, X):
        return self.coef_ * X + self.intercept_

def get_model_info():
    """Get information about the current model"""
    if hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
        return {
            "type": "trained_linear_regression",
            "coefficient": float(model.coef_[0]) if hasattr(model.coef_, '__getitem__') else float(model.coef_),
            "intercept": float(model.intercept_),
            "source": "trained_model" if os.path.exists('model.pkl') else "fallback_model"
        }
    else:
        return {
            "type": "unknown",
            "source": "unknown"
        }

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    db_status = "healthy" if wait_for_database() else "unhealthy"
    model_status = "loaded" if model is not None else "not_loaded"
    
    return jsonify({
        "status": "healthy",
        "database": db_status,
        "model": model_status,
        "model_info": get_model_info()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions using the trained model"""
    try:
        data = request.get_json()
        
        if not data or 'values' not in data:
            return jsonify({"error": "Please provide 'values' in the request body"}), 400
        
        values = data['values']
        
        if not isinstance(values, list):
            return jsonify({"error": "'values' must be a list of numbers"}), 400
        
        if not values:
            return jsonify({"error": "'values' list cannot be empty"}), 400
        
        # Convert to numpy array and reshape for prediction
        X = np.array(values).reshape(-1, 1)
        
        # Make predictions
        predictions = model.predict(X)
        
        # Convert predictions to list for JSON serialization
        result = {
            "predictions": predictions.tolist(),
            "model_info": get_model_info(),
            "input_values": values
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/data/sample', methods=['GET'])
def get_sample_data():
    """Get sample data from the database"""
    try:
        if not wait_for_database():
            return jsonify({"error": "Database not available"}), 503
        
        # Database connection parameters
        db_url = f"postgresql://{os.getenv('POSTGRES_USER', 'postgres')}:{os.getenv('POSTGRES_PASSWORD', 'postgres')}@{os.getenv('DB_HOST', 'database')}:{os.getenv('DB_PORT', '5432')}/{os.getenv('POSTGRES_DB', 'dsproject')}"
        
        engine = create_engine(db_url)
        
        # Get sample data
        query = "SELECT x, y FROM linear_data ORDER BY x LIMIT 10"
        with engine.connect() as conn:
            result = conn.execute(text(query))
            sample_data = [{"x": float(row[0]), "y": float(row[1])} for row in result]
        
        return jsonify({
            "sample_data": sample_data,
            "total_records": len(sample_data)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with usage instructions"""
    return jsonify({
        "message": "Multi-Container Data Pipeline - Model Server",
        "endpoints": {
            "GET /": "This help message",
            "GET /health": "Health check with database and model status",
            "POST /predict": "Make predictions (send JSON with 'values' array)",
            "GET /data/sample": "Get sample data from database"
        },
        "example": {
            "POST /predict": {
                "values": [1, 2, 3, 4, 5]
            }
        },
        "model_status": get_model_info()
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting model server on port {port}...")
    
    # Initialize before starting
    print("🚀 Initializing model server...")
    
    # Wait for database and load model
    if wait_for_database():
        load_model()
    else:
        print("⚠️  Database not available, using fallback model")
        load_model()
    
    app.run(host='0.0.0.0', port=port, debug=False)
