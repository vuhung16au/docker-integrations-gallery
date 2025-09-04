# Heart Disease Prediction Project

A comprehensive machine learning project for predicting heart disease using multiple algorithms and a complete ML pipeline.

## 🏥 Project Overview

This project implements a complete machine learning pipeline for heart disease prediction using the UCI Heart Disease dataset. It includes data preparation, exploratory data analysis, model training with multiple algorithms, and a Flask web service for predictions.

## 📊 Dataset

- **Source**: UCI Machine Learning Repository - Heart Disease Dataset
- **Size**: 303 records (296 after cleaning)
- **Features**: 13 medical indicators
- **Target**: Binary classification (0: No disease, 1: Heart disease)

### Features

- **age**: Age in years
- **sex**: Gender (0: Female, 1: Male)
- **cp**: Chest pain type (0-3)
- **trestbps**: Resting blood pressure (mm Hg)
- **chol**: Serum cholesterol (mg/dl)
- **fbs**: Fasting blood sugar > 120 mg/dl (0/1)
- **restecg**: Resting ECG results (0-2)
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise induced angina (0/1)
- **oldpeak**: ST depression induced by exercise
- **slope**: Slope of peak exercise ST segment (0-2)
- **ca**: Number of major vessels (0-3)
- **thal**: Thalassemia (1-3)

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker (optional)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd docker-integrations-gallery/05.Heart-Disease-Predictions
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the complete pipeline**
   ```bash
   python 80-run_pipeline.py
   ```

### Using Docker

1. **Build the Docker image**
   ```bash
   docker build -t heart-disease-prediction .
   ```

2. **Run the container**
   ```bash
   docker run -p 5000:5000 heart-disease-prediction
   ```

3. **Using docker-compose**
   ```bash
   docker-compose up -d
   ```

## 📁 Project Structure

```
05.Heart-Disease-Predictions/
├── Dataset-Heart/           # Original dataset
├── docs/                    # Documentation and visualizations
├── models/                  # Trained ML models
├── processed_data/          # Preprocessed data
├── 00-prepare_data.py      # Data preparation script
├── 20-exploratory_analysis.py  # EDA script
├── 40-train_models.py      # Model training script
├── 60-evaluate_models.py   # Model evaluation script
├── 80-run_pipeline.py      # Main pipeline script
├── app.py                   # Flask web service
├── test-endpoints.sh        # Endpoint testing script
├── requirements.txt         # Python dependencies
├── Dockerfile              # Docker configuration
├── docker-compose.yaml     # Docker Compose configuration
└── README.md               # This file
```

## 🔧 Pipeline Steps

### 1. Data Preparation (`00-prepare_data.py`)
- Load and clean the dataset
- Remove faulty data points
- Rename columns for clarity
- Split data into training/testing sets

### 2. Exploratory Data Analysis (`20-exploratory_analysis.py`)
- Target distribution analysis
- Numerical and categorical feature analysis
- Correlation analysis
- Feature importance visualization

### 3. Model Training (`40-train_models.py`)
- **Scikit-learn Models**:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine
  - Decision Tree
  - K-Nearest Neighbors
  - AdaBoost
  - Gradient Boosting
  - Naive Bayes
  - Linear/Quadratic Discriminant Analysis
  - Neural Network

- **Gradient Boosting Models**:
  - CatBoost
  - LightGBM
  - XGBoost

- Hyperparameter tuning for best models

### 4. Model Evaluation (`60-evaluate_models.py`)
- Performance metrics comparison
- ROC curves analysis
- Confusion matrices
- Feature importance analysis

## 🌐 Web Service

The project includes a Flask web service with the following endpoints:

### Health and Status
- `GET /health` - Health check
- `GET /status` - Model status
- `GET /sample` - Dataset sample

### Prediction Endpoints
- `POST /predict/logistic_regression` - Logistic Regression predictions
- `POST /predict/random_forest` - Random Forest predictions
- `POST /predict/svm` - Support Vector Machine predictions
- `POST /predict/decision_tree` - Decision Tree predictions
- `POST /predict/catboost` - CatBoost predictions
- `POST /predict/lightgbm` - LightGBM predictions
- `POST /predict/xgboost` - XGBoost predictions
- `POST /predict/all` - Predictions from all models

### Testing Endpoints

Use the included testing script to verify all endpoints:

```bash
# Make script executable (first time only)
chmod +x test-endpoints.sh

# Run endpoint tests
./test-endpoints.sh
```

The script will test:
- **Health & Status**: `/health`, `/status`, `/sample`
- **Scikit-learn Models**: All prediction endpoints
- **Gradient Boosting**: CatBoost, LightGBM, XGBoost
- **Multi-model**: `/predict/all`

### Example Prediction Request

```bash
curl -X POST http://localhost:5000/predict/logistic_regression \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1
  }'
```

## 📈 Results

The pipeline generates comprehensive reports and visualizations:

- **Model Performance Summary**: CSV file with all metrics
- **Performance Comparison Plots**: Bar charts comparing models
- **ROC Curves**: ROC analysis for all models
- **Confusion Matrices**: Grid of confusion matrices
- **Feature Importance**: Analysis of feature contributions
- **Pipeline Summary**: JSON summary of execution

## 🎯 Key Findings

Based on the analysis:
- **Most Predictive Features**: Number of major vessels, chest pain type, ST slope
- **Best Traditional ML**: Logistic Regression (~86.5% accuracy)
- **Best Gradient Boosting**: LightGBM (~86% accuracy after tuning)
- **Balanced Dataset**: 54% heart disease, 46% no disease

## 🛠️ Customization

### Adding New Models
1. Add model to the training script (`40-train_models.py`)
2. Update the Flask app (`app.py`) with new endpoints
3. Add model to requirements if needed

### Modifying Features
1. Update feature names in data preparation
2. Modify validation in Flask app
3. Retrain models with new feature set

### Hyperparameter Tuning
- Modify parameter grids in `40-train_models.py`
- Adjust RandomizedSearchCV parameters
- Add new tuning strategies

## 📝 Logging

The pipeline generates detailed logs:
- **Console Output**: Real-time progress updates
- **Pipeline Log**: Complete execution log (`pipeline.log`)
- **Model Logs**: Individual script execution logs

## 🐛 Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Dataset Not Found**
   - Ensure `Dataset-Heart/heart.csv` exists
   - Check file permissions

3. **Memory Issues**
   - Reduce batch sizes in training
   - Use smaller parameter grids for tuning

4. **Docker Issues**
   - Check Docker daemon is running
   - Verify port 5000 is available

5. **Endpoint Testing Issues**
   - Ensure Docker container is running
   - Check if port 5000 is accessible
   - Verify Flask service is healthy

### Debug Mode

Run individual scripts for debugging:
```bash
python 00-prepare_data.py
python 20-exploratory_analysis.py
python 40-train_models.py
python 60-evaluate_models.py
```

### Testing Endpoints

```bash
# Test individual endpoints
curl http://localhost:5000/health
curl http://localhost:5000/status
curl http://localhost:5000/sample

# Run comprehensive endpoint tests
./test-endpoints.sh
```

## 📚 References

- [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [CatBoost Documentation](https://catboost.ai/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

---

**Note**: This project is for educational and research purposes. Medical decisions should not be based solely on ML predictions. Always consult healthcare professionals for medical advice.
