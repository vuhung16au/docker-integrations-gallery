#!/usr/bin/env python3
"""
Heart Disease Prediction Service
Flask application providing multiple ML model endpoints for heart disease prediction
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables
MODELS = {}
DATASET_PATH = 'Dataset-Heart/heart.csv'
FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

def load_models():
    """Load all trained models"""
    models_dir = 'models'
    if not os.path.exists(models_dir):
        logger.warning("Models directory not found. Using mock models for demonstration.")
        create_mock_models()
        return
    
    try:
        # Load scikit-learn models
        model_files = {
            'logistic_regression': 'logistic_regression.pkl',
            'random_forest': 'random_forest.pkl',
            'svm': 'svm.pkl',
            'decision_tree': 'decision_tree.pkl',
            'catboost': 'catboost.pkl',
            'lightgbm': 'lightgbm.pkl',
            'xgboost': 'xgboost.pkl'
        }
        
        for model_name, filename in model_files.items():
            model_path = os.path.join(models_dir, filename)
            if os.path.exists(model_path):
                try:
                    MODELS[model_name] = joblib.load(model_path)
                    logger.info(f"Loaded model: {model_name}")
                except Exception as e:
                    logger.error(f"Error loading {model_name}: {e}")
                    create_mock_model(model_name)
            else:
                logger.warning(f"Model file not found: {model_path}")
                create_mock_model(model_name)
        
        if not MODELS:
            logger.warning("No models loaded. Creating mock models for demonstration.")
            create_mock_models()
        else:
            logger.info(f"Loaded {len(MODELS)} models successfully")
            
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        create_mock_models()

def create_mock_model(model_name):
    """Create a mock model for demonstration purposes"""
    class MockModel:
        def __init__(self, name):
            self.name = name
        
        def predict(self, X):
            # Return mock predictions (random 0 or 1)
            return np.random.choice([0, 1], size=X.shape[0])
        
        def predict_proba(self, X):
            # Return mock probabilities
            prob = np.random.uniform(0.3, 0.7, size=X.shape[0])
            return np.column_stack([1 - prob, prob])
    
    MODELS[model_name] = MockModel(model_name)
    logger.info(f"Created mock model: {model_name}")

def create_mock_models():
    """Create mock models for all required model types"""
    model_names = [
        'logistic_regression', 'random_forest', 'svm', 'decision_tree',
        'catboost', 'lightgbm', 'xgboost'
    ]
    
    for model_name in model_names:
        create_mock_model(model_name)

def validate_input(data):
    """Validate input data format and values"""
    try:
        # Check if all required features are present
        for feature in FEATURE_NAMES:
            if feature not in data:
                return False, f"Missing feature: {feature}"
        
        # Validate data types and ranges
        if not isinstance(data['age'], (int, float)) or data['age'] < 0 or data['age'] > 120:
            return False, "Invalid age value"
        
        if data['sex'] not in [0, 1]:
            return False, "Invalid sex value (must be 0 or 1)"
        
        if data['cp'] not in [0, 1, 2, 3]:
            return False, "Invalid chest pain type (must be 0-3)"
        
        if not isinstance(data['trestbps'], (int, float)) or data['trestbps'] < 50 or data['trestbps'] > 300:
            return False, "Invalid resting blood pressure value"
        
        if not isinstance(data['chol'], (int, float)) or data['chol'] < 100 or data['chol'] > 600:
            return False, "Invalid cholesterol value"
        
        if data['fbs'] not in [0, 1]:
            return False, "Invalid fasting blood sugar value (must be 0 or 1)"
        
        if data['restecg'] not in [0, 1, 2]:
            return False, "Invalid resting ECG value (must be 0-2)"
        
        if not isinstance(data['thalach'], (int, float)) or data['thalach'] < 50 or data['thalach'] > 250:
            return False, "Invalid max heart rate value"
        
        if data['exang'] not in [0, 1]:
            return False, "Invalid exercise induced angina value (must be 0 or 1)"
        
        if not isinstance(data['oldpeak'], (int, float)) or data['oldpeak'] < 0 or data['oldpeak'] > 10:
            return False, "Invalid ST depression value"
        
        if data['slope'] not in [0, 1, 2]:
            return False, "Invalid ST slope value (must be 0-2)"
        
        if data['ca'] not in [0, 1, 2, 3]:
            return False, "Invalid number of vessels value (must be 0-3)"
        
        if data['thal'] not in [1, 2, 3]:
            return False, "Invalid thalassemia value (must be 1-3)"
        
        return True, "Input validation passed"
    
    except Exception as e:
        return False, f"Validation error: {str(e)}"

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'heart-disease-prediction',
        'models_loaded': len(MODELS) > 0,
        'mock_models': any(hasattr(model, 'name') and 'Mock' in str(type(model)) for model in MODELS.values())
    })

@app.route('/status', methods=['GET'])
def model_status():
    """Get status of all models"""
    status = {}
    for model_name, model in MODELS.items():
        is_mock = hasattr(model, 'name') and 'Mock' in str(type(model))
        status[model_name] = {
            'loaded': model is not None,
            'type': type(model).__name__,
            'is_mock': is_mock
        }
    
    return jsonify({
        'total_models': len(MODELS),
        'models': status,
        'note': 'Mock models are used when trained models are not available'
    })

@app.route('/sample', methods=['GET'])
def get_data_sample():
    """Get a sample of the dataset"""
    try:
        df = pd.read_csv(DATASET_PATH)
        sample = df.head(5).to_dict('records')
        return jsonify({
            'sample_size': 5,
            'total_records': len(df),
            'features': FEATURE_NAMES,
            'sample_data': sample
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/logistic_regression', methods=['POST'])
def predict_logistic_regression():
    """Predict using Logistic Regression"""
    return predict_with_model('logistic_regression')

@app.route('/predict/random_forest', methods=['POST'])
def predict_random_forest():
    """Predict using Random Forest"""
    return predict_with_model('random_forest')

@app.route('/predict/svm', methods=['POST'])
def predict_svm():
    """Predict using Support Vector Machine"""
    return predict_with_model('svm')

@app.route('/predict/decision_tree', methods=['POST'])
def predict_decision_tree():
    """Predict using Decision Tree"""
    return predict_with_model('decision_tree')

@app.route('/predict/catboost', methods=['POST'])
def predict_catboost():
    """Predict using CatBoost"""
    return predict_with_model('catboost')

@app.route('/predict/lightgbm', methods=['POST'])
def predict_lightgbm():
    """Predict using LightGBM"""
    return predict_with_model('lightgbm')

@app.route('/predict/xgboost', methods=['POST'])
def predict_xgboost():
    """Predict using XGBoost"""
    return predict_with_model('xgboost')

def predict_with_model(model_name):
    """Generic prediction function for all models"""
    if model_name not in MODELS:
        return jsonify({'error': f'Model {model_name} not found'}), 404
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate input
        is_valid, message = validate_input(data)
        if not is_valid:
            return jsonify({'error': message}), 400
        
        # Prepare features
        features = [data[feature] for feature in FEATURE_NAMES]
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        model = MODELS[model_name]
        prediction = model.predict(features_array)[0]
        probability = model.predict_proba(features_array)[0].tolist() if hasattr(model, 'predict_proba') else None
        
        # Check if this is a mock model
        is_mock = hasattr(model, 'name') and 'Mock' in str(type(model))
        
        response = {
            'model': model_name,
            'prediction': int(prediction),
            'prediction_label': 'Heart Disease' if prediction == 1 else 'No Heart Disease',
            'probability': probability,
            'input_features': dict(zip(FEATURE_NAMES, features)),
            'is_mock_prediction': is_mock
        }
        
        if is_mock:
            response['note'] = 'This is a mock prediction. Train models for real predictions.'
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Prediction error with {model_name}: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/predict/all', methods=['POST'])
def predict_all_models():
    """Get predictions from all available models"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate input
        is_valid, message = validate_input(data)
        if not is_valid:
            return jsonify({'error': message}), 400
        
        # Prepare features
        features = [data[feature] for feature in FEATURE_NAMES]
        features_array = np.array(features).reshape(1, -1)
        
        results = {}
        for model_name, model in MODELS.items():
            try:
                prediction = model.predict(features_array)[0]
                probability = model.predict_proba(features_array)[0].tolist() if hasattr(model, 'predict_proba') else None
                
                is_mock = hasattr(model, 'name') and 'Mock' in str(type(model))
                
                results[model_name] = {
                    'prediction': int(prediction),
                    'prediction_label': 'Heart Disease' if prediction == 1 else 'No Heart Disease',
                    'probability': probability,
                    'is_mock_prediction': is_mock
                }
                
                if is_mock:
                    results[model_name]['note'] = 'Mock prediction'
                    
            except Exception as e:
                results[model_name] = {'error': str(e)}
        
        response = {
            'input_features': dict(zip(FEATURE_NAMES, features)),
            'predictions': results,
            'note': 'Mock models are used when trained models are not available'
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Multi-model prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    # Load models on startup
    load_models()
    
    # Run the application
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
