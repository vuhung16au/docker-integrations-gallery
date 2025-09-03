#!/usr/bin/env python3
"""
Simple Linear Regression Model Server
A minimalistic Flask server that serves a pre-trained linear regression model.
"""

from flask import Flask, request, jsonify
import numpy as np
import pickle
import os

app = Flask(__name__)

# Simple linear regression model: y = 2x + 1
class SimpleLinearModel:
    def __init__(self):
        self.coef_ = 2.0
        self.intercept_ = 1.0
    
    def predict(self, X):
        return self.coef_ * X + self.intercept_

# Initialize model
model = SimpleLinearModel()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model": "linear_regression"})

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions using the linear regression model"""
    try:
        data = request.get_json()
        
        if not data or 'values' not in data:
            return jsonify({"error": "Please provide 'values' in the request body"}), 400
        
        values = data['values']
        
        if not isinstance(values, list):
            return jsonify({"error": "'values' must be a list of numbers"}), 400
        
        # Convert to numpy array and reshape for prediction
        X = np.array(values).reshape(-1, 1)
        
        # Make predictions
        predictions = model.predict(X)
        
        # Convert predictions to list for JSON serialization
        result = {
            "predictions": predictions.tolist(),
            "model_info": {
                "type": "linear_regression",
                "equation": "y = 2x + 1",
                "coefficient": model.coef_,
                "intercept": model.intercept_
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with usage instructions"""
    return jsonify({
        "message": "Linear Regression Model Server",
        "endpoints": {
            "GET /": "This help message",
            "GET /health": "Health check",
            "POST /predict": "Make predictions (send JSON with 'values' array)"
        },
        "example": {
            "POST /predict": {
                "values": [1, 2, 3, 4, 5]
            }
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
