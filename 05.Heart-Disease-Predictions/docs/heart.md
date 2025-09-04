# Heart Disease Prediction Dataset Documentation

## Overview
This project uses the Heart Disease dataset from the UCI Machine Learning Repository to predict the presence of heart disease based on various medical indicators.

## Dataset Information
- **Source**: UCI Machine Learning Repository
- **Original Dataset**: Cleveland Heart Disease Database
- **Size**: 303 records (296 after cleaning)
- **Features**: 13 independent variables
- **Target**: 1 binary variable (heart disease presence)

## Data Dictionary

### Target Variable
- **target**: Diagnosis of heart disease (angiographic disease status)
  - 0: < 50% diameter narrowing (no disease)
  - 1: > 50% diameter narrowing (disease)

### Independent Variables

#### Numerical Features
1. **age**: Age in years (29-77)
2. **trestbps**: Resting blood pressure in mm Hg (94-200)
3. **chol**: Serum cholesterol in mg/dl (126-564)
4. **thalach**: Maximum heart rate achieved (71-202)
5. **oldpeak**: ST depression induced by exercise relative to rest (0.0-6.2)
6. **ca**: Number of major vessels colored by fluoroscopy (0-3)

#### Categorical Features
1. **sex**: Gender
   - 0: Female
   - 1: Male

2. **cp**: Chest pain type
   - 0: Typical angina
   - 1: Atypical angina
   - 2: Non-anginal pain
   - 3: Asymptomatic

3. **fbs**: Fasting blood sugar > 120 mg/dl
   - 0: False
   - 1: True

4. **restecg**: Resting electrocardiographic results
   - 0: Normal
   - 1: ST-T wave abnormality
   - 2: Left ventricular hypertrophy

5. **exang**: Exercise induced angina
   - 0: No
   - 1: Yes

6. **slope**: Slope of peak exercise ST segment
   - 0: Upsloping
   - 1: Flat
   - 2: Downsloping

7. **thal**: Thalassemia
   - 0: Error (originally NaN)
   - 1: Fixed defect
   - 2: Normal
   - 3: Reversible defect

## Data Quality Notes
- 7 records were removed due to faulty data (ca=4, thal=0)
- No missing values in the cleaned dataset
- Target variable is fairly balanced (54% disease, 46% no disease)

## Key Insights from EDA
1. **Most Predictive Features**:
   - Number of major vessels (ca)
   - Chest pain type (cp)
   - ST slope (slope)
   - Maximum heart rate achieved (thalach)
   - ST depression (oldpeak)

2. **Correlations**:
   - Weak correlations between most features
   - Stronger correlations with target: ca (-0.47), thalach (0.43), oldpeak (-0.43)

3. **Medical Significance**:
   - Chest pain patterns are strong indicators
   - Exercise test results (ST changes, heart rate) are predictive
   - Vessel blockage (ca) is highly indicative

## Model Performance
- **Best Traditional ML**: Logistic Regression (86.5% accuracy)
- **Best Gradient Boosting**: LightGBM (86% accuracy after tuning)
- **Key Metrics**: Balanced precision and recall for medical applications

## Usage Notes
- Dataset is suitable for binary classification
- Features are already preprocessed and encoded
- Medical consultation recommended for clinical applications
- Model should be used as screening tool, not diagnostic
