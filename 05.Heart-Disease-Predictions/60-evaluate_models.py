#!/usr/bin/env python3
"""
Model Evaluation Script for Heart Disease Prediction
Evaluates and compares all trained models with detailed metrics and visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_models_and_data():
    """Load all trained models and test data"""
    try:
        # Load test data
        X_test = np.load('processed_data/X_test.npy')
        y_test = np.load('processed_data/y_test.npy')
        
        # Load feature names
        with open('processed_data/feature_names.txt', 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        
        # Load all models
        models_dir = 'models'
        models = {}
        
        for filename in os.listdir(models_dir):
            if filename.endswith('.pkl'):
                model_name = filename.replace('.pkl', '')
                model_path = os.path.join(models_dir, filename)
                try:
                    models[model_name] = joblib.load(model_path)
                    logger.info(f"Loaded model: {model_name}")
                except Exception as e:
                    logger.error(f"Error loading {model_name}: {e}")
        
        logger.info(f"Loaded {len(models)} models and test data: {X_test.shape}")
        return models, X_test, y_test, feature_names
        
    except Exception as e:
        logger.error(f"Error loading models and data: {e}")
        raise

def evaluate_single_model(model, model_name, X_test, y_test):
    """Evaluate a single model and return comprehensive metrics"""
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # ROC AUC (if probability predictions available)
        roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
        
        # Average precision score
        avg_precision = average_precision_score(y_test, y_pred_proba) if y_pred_proba is not None else None
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'average_precision': avg_precision,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'confusion_matrix': cm,
            'classification_report': report
        }
        
        logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error evaluating {model_name}: {e}")
        return None

def evaluate_all_models(models, X_test, y_test):
    """Evaluate all models and return results"""
    logger.info("Evaluating all models...")
    
    results = {}
    
    for model_name, model in models.items():
        result = evaluate_single_model(model, model_name, X_test, y_test)
        if result:
            results[model_name] = result
    
    logger.info(f"Evaluation completed for {len(results)} models")
    return results

def create_performance_comparison_plot(evaluation_results):
    """Create performance comparison plots"""
    logger.info("Creating performance comparison plots...")
    
    # Extract metrics for plotting
    model_names = list(evaluation_results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics):
        values = [evaluation_results[name][metric] for name in model_names]
        
        # Create bar plot
        bars = axes[i].bar(range(len(model_names)), values, color='skyblue', alpha=0.7)
        axes[i].set_title(f'{metric.replace("_", " ").title()} Comparison')
        axes[i].set_xlabel('Models')
        axes[i].set_ylabel(metric.replace("_", " ").title())
        axes[i].set_xticks(range(len(model_names)))
        axes[i].set_xticklabels(model_names, rotation=45, ha='right')
        axes[i].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('docs/model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Performance comparison plots created and saved")

def create_roc_curves_plot(evaluation_results):
    """Create ROC curves comparison plot"""
    logger.info("Creating ROC curves comparison plot...")
    
    plt.figure(figsize=(10, 8))
    
    for model_name, result in evaluation_results.items():
        if result['probabilities'] is not None:
            fpr, tpr, _ = roc_curve(result['y_test'], result['probabilities'])
            roc_auc = result['roc_auc']
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('docs/roc_curves_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("ROC curves comparison plot created and saved")

def create_confusion_matrices_grid(evaluation_results):
    """Create confusion matrices grid for all models"""
    logger.info("Creating confusion matrices grid...")
    
    n_models = len(evaluation_results)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, (model_name, result) in enumerate(evaluation_results.items()):
        row = i // n_cols
        col = i % n_cols
        
        # Create confusion matrix heatmap
        cm = result['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[row, col])
        axes[row, col].set_title(f'{model_name}\nConfusion Matrix')
        axes[row, col].set_xlabel('Predicted')
        axes[row, col].set_ylabel('Actual')
    
    # Remove empty subplots
    for i in range(n_models, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        fig.delaxes(axes[row, col])
    
    plt.tight_layout()
    plt.savefig('docs/confusion_matrices_grid.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Confusion matrices grid created and saved")

def create_feature_importance_analysis(evaluation_results, feature_names):
    """Analyze feature importance for tree-based models"""
    logger.info("Analyzing feature importance...")
    
    # Models that typically have feature importance
    tree_models = ['random_forest', 'decision_tree', 'adaboost', 'gradient_boosting', 
                   'catboost', 'lightgbm', 'xgboost']
    
    feature_importance_data = {}
    
    for model_name in tree_models:
        if model_name in evaluation_results:
            model = evaluation_results[model_name]['model']
            
            try:
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feature_importance_data[model_name] = importances
                    logger.info(f"Feature importance extracted for {model_name}")
            except Exception as e:
                logger.warning(f"Could not extract feature importance for {model_name}: {e}")
    
    if feature_importance_data:
        # Create feature importance comparison plot
        n_models = len(feature_importance_data)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, (model_name, importances) in enumerate(feature_importance_data.items()):
            if i < len(axes):
                # Sort features by importance
                sorted_idx = np.argsort(importances)[::-1]
                sorted_features = [feature_names[j] for j in sorted_idx]
                sorted_importances = importances[sorted_idx]
                
                # Create bar plot
                axes[i].barh(range(len(sorted_features)), sorted_importances)
                axes[i].set_yticks(range(len(sorted_features)))
                axes[i].set_yticklabels(sorted_features)
                axes[i].set_xlabel('Feature Importance')
                axes[i].set_title(f'{model_name.title()} - Feature Importance')
                axes[i].grid(axis='x', alpha=0.3)
        
        # Remove extra subplots
        for i in range(len(feature_importance_data), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig('docs/feature_importance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Feature importance analysis completed and saved")
    
    return feature_importance_data

def generate_evaluation_report(evaluation_results, feature_importance_data):
    """Generate comprehensive evaluation report"""
    logger.info("Generating evaluation report...")
    
    # Create summary DataFrame
    summary_data = []
    for model_name, result in evaluation_results.items():
        summary_data.append({
            'Model': model_name,
            'Accuracy': f"{result['accuracy']:.4f}",
            'Precision': f"{result['precision']:.4f}",
            'Recall': f"{result['recall']:.4f}",
            'F1 Score': f"{result['f1_score']:.4f}",
            'ROC AUC': f"{result['roc_auc']:.4f}" if result['roc_auc'] else 'N/A',
            'Avg Precision': f"{result['average_precision']:.4f}" if result['average_precision'] else 'N/A'
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Sort by accuracy
    summary_df['Accuracy'] = pd.to_numeric(summary_df['Accuracy'])
    summary_df = summary_df.sort_values('Accuracy', ascending=False)
    
    # Save summary
    summary_df.to_csv('docs/model_evaluation_summary.csv', index=False)
    
    # Print summary
    logger.info("\n" + "="*100)
    logger.info("MODEL EVALUATION SUMMARY")
    logger.info("="*100)
    logger.info(f"\n{summary_df.to_string(index=False)}")
    
    # Find best models by different metrics
    best_accuracy = summary_df.iloc[0]
    logger.info(f"\nBest model by accuracy: {best_accuracy['Model']} ({best_accuracy['Accuracy']})")
    
    # Find best model by F1 score
    f1_scores = summary_df[summary_df['F1 Score'] != 'N/A'].copy()
    if not f1_scores.empty:
        f1_scores['F1 Score'] = pd.to_numeric(f1_scores['F1 Score'])
        best_f1 = f1_scores.loc[f1_scores['F1 Score'].idxmax()]
        logger.info(f"Best model by F1 score: {best_f1['Model']} ({best_f1['F1 Score']})")
    
    # Find best model by ROC AUC
    roc_auc_scores = summary_df[summary_df['ROC AUC'] != 'N/A'].copy()
    if not roc_auc_scores.empty:
        roc_auc_scores['ROC AUC'] = pd.to_numeric(roc_auc_scores['ROC AUC'])
        best_roc = roc_auc_scores.loc[roc_auc_scores['ROC AUC'].idxmax()]
        logger.info(f"Best model by ROC AUC: {best_roc['Model']} ({best_roc['ROC AUC']})")
    
    return summary_df

def main():
    """Main function to run the model evaluation pipeline"""
    try:
        logger.info("Starting model evaluation pipeline...")
        
        # Load models and data
        models, X_test, y_test, feature_names = load_models_and_data()
        
        # Evaluate all models
        evaluation_results = evaluate_all_models(models, X_test, y_test)
        
        # Create visualizations
        create_performance_comparison_plot(evaluation_results)
        create_roc_curves_plot(evaluation_results)
        create_confusion_matrices_grid(evaluation_results)
        
        # Analyze feature importance
        feature_importance_data = create_feature_importance_analysis(evaluation_results, feature_names)
        
        # Generate evaluation report
        evaluation_summary = generate_evaluation_report(evaluation_results, feature_importance_data)
        
        logger.info("Model evaluation pipeline completed successfully!")
        logger.info("All visualizations and reports saved to docs/ directory")
        
        return evaluation_results, evaluation_summary
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()
