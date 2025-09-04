#!/usr/bin/env python3
"""
Main Pipeline Script for Heart Disease Prediction
Runs the complete pipeline from data preparation to model deployment
"""

import os
import sys
import logging
import time
from datetime import datetime
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_script(script_name, description):
    """Run a Python script and return success status"""
    logger.info(f"Running {description}...")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            check=True
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"{description} completed successfully in {elapsed_time:.2f} seconds")
        logger.info(f"Output: {result.stdout}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        logger.error(f"{description} failed after {elapsed_time:.2f} seconds")
        logger.error(f"Error: {e.stderr}")
        return False
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"{description} failed after {elapsed_time:.2f} seconds")
        logger.error(f"Unexpected error: {e}")
        return False

def check_prerequisites():
    """Check if all prerequisites are met"""
    logger.info("Checking prerequisites...")
    
    # Check if dataset exists
    if not os.path.exists('Dataset-Heart/heart.csv'):
        logger.error("Dataset not found at Dataset-Heart/heart.csv")
        return False
    
    # Check if required directories exist
    required_dirs = ['Dataset-Heart', 'docs']
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
            logger.info(f"Created directory: {dir_name}")
    
    # Check if required Python packages are available
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn',
        'catboost', 'lightgbm', 'xgboost', 'flask', 'joblib'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        logger.error("Please install them using: pip install -r requirements.txt")
        return False
    
    logger.info("All prerequisites met!")
    return True

def run_data_preparation():
    """Run data preparation pipeline"""
    return run_script('00-prepare_data.py', 'Data Preparation')

def run_exploratory_analysis():
    """Run exploratory data analysis"""
    return run_script('20-exploratory_analysis.py', 'Exploratory Data Analysis')

def run_model_training():
    """Run model training pipeline"""
    return run_script('40-train_models.py', 'Model Training')

def run_model_evaluation():
    """Run model evaluation pipeline"""
    return run_script('60-evaluate_models.py', 'Model Evaluation')

def create_pipeline_summary():
    """Create a summary of the pipeline execution"""
    logger.info("Creating pipeline summary...")
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'pipeline_version': '1.0.0',
        'dataset': 'Heart Disease Prediction Dataset',
        'total_records': 0,
        'features': 0,
        'models_trained': 0,
        'best_model': None,
        'best_accuracy': 0.0
    }
    
    # Count dataset records
    try:
        import pandas as pd
        df = pd.read_csv('Dataset-Heart/heart.csv')
        summary['total_records'] = len(df)
        summary['features'] = len(df.columns) - 1  # Exclude target
    except Exception as e:
        logger.warning(f"Could not count dataset records: {e}")
    
    # Count trained models
    models_dir = 'models'
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
        summary['models_trained'] = len(model_files)
    
    # Find best model performance
    try:
        if os.path.exists('docs/model_evaluation_summary.csv'):
            import pandas as pd
            perf_df = pd.read_csv('docs/model_evaluation_summary.csv')
            if not perf_df.empty:
                best_idx = perf_df['Accuracy'].astype(float).idxmax()
                summary['best_model'] = perf_df.loc[best_idx, 'Model']
                summary['best_accuracy'] = float(perf_df.loc[best_idx, 'Accuracy'])
    except Exception as e:
        logger.warning(f"Could not read performance summary: {e}")
    
    # Save summary
    import json
    with open('docs/pipeline_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info("="*60)
    logger.info(f"Timestamp: {summary['timestamp']}")
    logger.info(f"Dataset Records: {summary['total_records']}")
    logger.info(f"Features: {summary['features']}")
    logger.info(f"Models Trained: {summary['models_trained']}")
    logger.info(f"Best Model: {summary['best_model']}")
    logger.info(f"Best Accuracy: {summary['best_accuracy']:.4f}")
    logger.info("="*60)
    
    return summary

def main():
    """Main function to run the complete pipeline"""
    start_time = time.time()
    
    logger.info("="*60)
    logger.info("HEART DISEASE PREDICTION PIPELINE")
    logger.info("="*60)
    logger.info(f"Pipeline started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Check prerequisites
        if not check_prerequisites():
            logger.error("Prerequisites check failed. Exiting pipeline.")
            return False
        
        # Step 1: Data Preparation
        logger.info("\n" + "="*40)
        logger.info("STEP 1: DATA PREPARATION")
        logger.info("="*40)
        if not run_data_preparation():
            logger.error("Data preparation failed. Exiting pipeline.")
            return False
        
        # Step 2: Exploratory Data Analysis
        logger.info("\n" + "="*40)
        logger.info("STEP 2: EXPLORATORY DATA ANALYSIS")
        logger.info("="*40)
        if not run_exploratory_analysis():
            logger.error("Exploratory data analysis failed. Exiting pipeline.")
            return False
        
        # Step 3: Model Training
        logger.info("\n" + "="*40)
        logger.info("STEP 3: MODEL TRAINING")
        logger.info("="*40)
        if not run_model_training():
            logger.error("Model training failed. Exiting pipeline.")
            return False
        
        # Step 4: Model Evaluation
        logger.info("\n" + "="*40)
        logger.info("STEP 4: MODEL EVALUATION")
        logger.info("="*40)
        if not run_model_evaluation():
            logger.error("Model evaluation failed. Exiting pipeline.")
            return False
        
        # Create pipeline summary
        logger.info("\n" + "="*40)
        logger.info("GENERATING PIPELINE SUMMARY")
        logger.info("="*40)
        summary = create_pipeline_summary()
        
        # Pipeline completed successfully
        total_time = time.time() - start_time
        logger.info("\n" + "="*60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"Total execution time: {total_time:.2f} seconds")
        logger.info(f"All outputs saved to docs/ directory")
        logger.info(f"Models saved to models/ directory")
        logger.info("="*60)
        
        return True
        
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"\nPipeline failed after {total_time:.2f} seconds")
        logger.error(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
