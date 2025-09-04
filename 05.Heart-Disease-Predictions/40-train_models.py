#!/usr/bin/env python3
"""
Model Training Script for Heart Disease Prediction
Trains multiple ML models including scikit-learn and gradient boosting models
"""

import numpy as np
import pandas as pd
import joblib
import os
import logging
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import catboost as cb
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_training_data():
    """Load the prepared training data"""
    try:
        X_train = np.load('processed_data/X_train.npy')
        X_test = np.load('processed_data/X_test.npy')
        y_train = np.load('processed_data/y_train.npy')
        y_test = np.load('processed_data/y_test.npy')
        
        with open('processed_data/feature_names.txt', 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        
        logger.info(f"Training data loaded - X_train: {X_train.shape}, X_test: {X_test.shape}")
        logger.info(f"Feature names: {feature_names}")
        
        return X_train, X_test, y_train, y_test, feature_names
        
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        raise

def train_scikit_learn_models(X_train, y_train, X_test, y_test):
    """Train scikit-learn models"""
    logger.info("Training scikit-learn models...")
    
    models = {
        'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
        'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'decision_tree': DecisionTreeClassifier(random_state=42),
        'svm': SVC(random_state=42, probability=True),
        'nu_svc': NuSVC(random_state=42, probability=True),
        'knn': KNeighborsClassifier(n_neighbors=5),
        'adaboost': AdaBoostClassifier(random_state=42),
        'gradient_boosting': GradientBoostingClassifier(random_state=42),
        'naive_bayes': GaussianNB(),
        'linear_da': LinearDiscriminantAnalysis(),
        'quadratic_da': QuadraticDiscriminantAnalysis(),
        'neural_net': MLPClassifier(random_state=42, max_iter=1000, hidden_layer_sizes=(100, 50))
    }
    
    results = {}
    
    for name, model in models.items():
        try:
            logger.info(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
            
            # Classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1_score': report['weighted avg']['f1-score']
            }
            
            logger.info(f"{name} - Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f if roc_auc else 'N/A'}")
            
        except Exception as e:
            logger.error(f"Error training {name}: {e}")
            results[name] = {'error': str(e)}
    
    return results

def train_gradient_boosting_models(X_train, y_train, X_test, y_test):
    """Train gradient boosting models"""
    logger.info("Training gradient boosting models...")
    
    models = {
        'catboost': cb.CatBoostClassifier(random_state=42, verbose=False),
        'lightgbm': lgb.LGBMClassifier(random_state=42, verbose=-1),
        'xgboost': xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    results = {}
    
    for name, model in models.items():
        try:
            logger.info(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            # Classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1_score': report['weighted avg']['f1-score']
            }
            
            logger.info(f"{name} - Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}")
            
        except Exception as e:
            logger.error(f"Error training {name}: {e}")
            results[name] = {'error': str(e)}
    
    return results

def hyperparameter_tuning_best_models(X_train, y_train, X_test, y_test, scikit_results, gb_results):
    """Perform hyperparameter tuning on the best performing models"""
    logger.info("Performing hyperparameter tuning on best models...")
    
    # Find best scikit-learn model
    best_scikit = None
    best_scikit_score = 0
    for name, result in scikit_results.items():
        if 'error' not in result and result['accuracy'] > best_scikit_score:
            best_scikit = name
            best_scikit_score = result['accuracy']
    
    # Find best gradient boosting model
    best_gb = None
    best_gb_score = 0
    for name, result in gb_results.items():
        if 'error' not in result and result['accuracy'] > best_gb_score:
            best_gb = name
            best_gb_score = result['accuracy']
    
    tuned_models = {}
    
    # Tune best scikit-learn model
    if best_scikit:
        logger.info(f"Tuning {best_scikit}...")
        try:
            if best_scikit == 'logistic_regression':
                param_grid = {
                    'C': np.logspace(-4, 4, 20),
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
                model = LogisticRegression(random_state=42, max_iter=1000)
                
            elif best_scikit == 'random_forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                model = RandomForestClassifier(random_state=42)
            
            # Perform randomized search
            random_search = RandomizedSearchCV(
                model, param_grid, n_iter=20, cv=5, scoring='accuracy', 
                random_state=42, n_jobs=-1
            )
            random_search.fit(X_train, y_train)
            
            # Evaluate tuned model
            y_pred = random_search.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            tuned_models[f'{best_scikit}_tuned'] = {
                'model': random_search.best_estimator_,
                'accuracy': accuracy,
                'best_params': random_search.best_params_
            }
            
            logger.info(f"{best_scikit} tuned - Best accuracy: {accuracy:.4f}")
            logger.info(f"Best parameters: {random_search.best_params_}")
            
        except Exception as e:
            logger.error(f"Error tuning {best_scikit}: {e}")
    
    # Tune best gradient boosting model
    if best_gb:
        logger.info(f"Tuning {best_gb}...")
        try:
            if best_gb == 'lightgbm':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7, -1],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'num_leaves': [31, 50, 100]
                }
                model = lgb.LGBMClassifier(random_state=42, verbose=-1)
                
            elif best_gb == 'catboost':
                param_grid = {
                    'iterations': [100, 200, 300],
                    'depth': [4, 6, 8],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'l2_leaf_reg': [1, 3, 5]
                }
                model = cb.CatBoostClassifier(random_state=42, verbose=False)
                
            elif best_gb == 'xgboost':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                }
                model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
            
            # Perform randomized search
            random_search = RandomizedSearchCV(
                model, param_grid, n_iter=20, cv=5, scoring='accuracy', 
                random_state=42, n_jobs=-1
            )
            random_search.fit(X_train, y_train)
            
            # Evaluate tuned model
            y_pred = random_search.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            tuned_models[f'{best_gb}_tuned'] = {
                'model': random_search.best_estimator_,
                'accuracy': accuracy,
                'best_params': random_search.best_params_
            }
            
            logger.info(f"{best_gb} tuned - Best accuracy: {accuracy:.4f}")
            logger.info(f"Best parameters: {random_search.best_params_}")
            
        except Exception as e:
            logger.error(f"Error tuning {best_gb}: {e}")
    
    return tuned_models

def save_models(all_results, tuned_results):
    """Save all trained models"""
    logger.info("Saving trained models...")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save scikit-learn models
    for name, result in all_results.items():
        if 'error' not in result:
            try:
                model_path = f'models/{name}.pkl'
                joblib.dump(result['model'], model_path)
                logger.info(f"Saved {name} to {model_path}")
            except Exception as e:
                logger.error(f"Error saving {name}: {e}")
    
    # Save tuned models
    for name, result in tuned_results.items():
        if 'error' not in result:
            try:
                model_path = f'models/{name}.pkl'
                joblib.dump(result['model'], model_path)
                logger.info(f"Saved {name} to {model_path}")
            except Exception as e:
                logger.error(f"Error saving {name}: {e}")
    
    logger.info("All models saved successfully!")

def generate_performance_summary(all_results, tuned_results):
    """Generate performance summary of all models"""
    logger.info("Generating performance summary...")
    
    # Combine all results
    all_models = {**all_results, **tuned_results}
    
    # Create performance summary
    summary_data = []
    for name, result in all_models.items():
        if 'error' not in result:
            summary_data.append({
                'Model': name,
                'Accuracy': f"{result['accuracy']:.4f}",
                'ROC AUC': f"{result.get('roc_auc', 'N/A')}",
                'Precision': f"{result.get('precision', 'N/A')}",
                'Recall': f"{result.get('recall', 'N/A')}",
                'F1 Score': f"{result.get('f1_score', 'N/A')}"
            })
    
    # Create DataFrame and sort by accuracy
    summary_df = pd.DataFrame(summary_data)
    summary_df['Accuracy'] = pd.to_numeric(summary_df['Accuracy'])
    summary_df = summary_df.sort_values('Accuracy', ascending=False)
    
    # Save summary
    summary_df.to_csv('docs/model_performance_summary.csv', index=False)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("MODEL PERFORMANCE SUMMARY")
    logger.info("="*80)
    logger.info(f"\n{summary_df.to_string(index=False)}")
    
    # Find best model
    best_model = summary_df.iloc[0]
    logger.info(f"\nBest performing model: {best_model['Model']} with accuracy: {best_model['Accuracy']}")
    
    return summary_df

def main():
    """Main function to run the model training pipeline"""
    try:
        logger.info("Starting model training pipeline...")
        
        # Load training data
        X_train, X_test, y_train, y_test, feature_names = load_training_data()
        
        # Train scikit-learn models
        scikit_results = train_scikit_learn_models(X_train, y_train, X_test, y_test)
        
        # Train gradient boosting models
        gb_results = train_gradient_boosting_models(X_train, y_train, X_test, y_test)
        
        # Combine all results
        all_results = {**scikit_results, **gb_results}
        
        # Hyperparameter tuning
        tuned_results = hyperparameter_tuning_best_models(X_train, y_train, X_test, y_test, scikit_results, gb_results)
        
        # Save models
        save_models(all_results, tuned_results)
        
        # Generate performance summary
        performance_summary = generate_performance_summary(all_results, tuned_results)
        
        logger.info("Model training pipeline completed successfully!")
        
        return all_results, tuned_results, performance_summary
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise

if __name__ == "__main__":
    main()
