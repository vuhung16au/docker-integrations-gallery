#!/usr/bin/env python3
"""
Exploratory Data Analysis for Heart Disease Prediction
Analyzes the dataset and creates visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_processed_data():
    """Load the processed data for analysis"""
    try:
        df = pd.read_csv('processed_data/heart_disease_processed.csv')
        logger.info(f"Processed data loaded. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading processed data: {e}")
        raise

def analyze_target_distribution(df):
    """Analyze the distribution of the target variable"""
    target_counts = df['target'].value_counts()
    target_percentages = df['target'].value_counts(normalize=True) * 100
    
    logger.info("Target Distribution Analysis:")
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"No disease (0): {target_counts[0]} ({target_percentages[0]:.1f}%)")
    logger.info(f"Heart disease (1): {target_counts[1]} ({target_percentages[1]:.1f}%)")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Count plot
    sns.countplot(data=df, x='target', ax=ax1)
    ax1.set_title('Target Distribution')
    ax1.set_xlabel('Target (0: No Disease, 1: Heart Disease)')
    ax1.set_ylabel('Count')
    
    # Pie chart
    ax2.pie(target_counts.values, labels=['No Disease', 'Heart Disease'], 
            autopct='%1.1f%%', startangle=90)
    ax2.set_title('Target Distribution (%)')
    
    plt.tight_layout()
    plt.savefig('docs/target_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return target_counts, target_percentages

def analyze_numerical_features(df):
    """Analyze numerical features"""
    numerical_features = ['age', 'resting_blood_pressure', 'cholesterol', 
                         'max_heart_rate_achieved', 'st_depression', 'num_major_vessels']
    
    logger.info("Numerical Features Analysis:")
    
    # Statistical summary
    numerical_df = df[numerical_features]
    stats_summary = numerical_df.describe()
    logger.info(f"\nStatistical Summary:\n{stats_summary}")
    
    # Create subplots for each numerical feature
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, feature in enumerate(numerical_features):
        # Distribution plot
        sns.histplot(data=df, x=feature, hue='target', bins=20, ax=axes[i], alpha=0.7)
        axes[i].set_title(f'Distribution of {feature.replace("_", " ").title()}')
        axes[i].set_xlabel(feature.replace("_", " ").title())
        axes[i].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('docs/numerical_features_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Correlation with target
    correlations = {}
    for feature in numerical_features:
        correlation = df[feature].corr(df['target'])
        correlations[feature] = correlation
        logger.info(f"{feature}: correlation with target = {correlation:.3f}")
    
    return stats_summary, correlations

def analyze_categorical_features(df):
    """Analyze categorical features"""
    categorical_features = ['sex', 'chest_pain_type', 'fasting_blood_sugar', 
                           'resting_electrocardiogram', 'exercise_induced_angina', 
                           'st_slope', 'thalassemia']
    
    logger.info("Categorical Features Analysis:")
    
    # Create subplots for each categorical feature
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    axes = axes.ravel()
    
    for i, feature in enumerate(categorical_features):
        if i < len(axes):
            # Count plot with target
            sns.countplot(data=df, x=feature, hue='target', ax=axes[i])
            axes[i].set_title(f'{feature.replace("_", " ").title()} vs Target')
            axes[i].set_xlabel(feature.replace("_", " ").title())
            axes[i].set_ylabel('Count')
            axes[i].tick_params(axis='x', rotation=45)
    
    # Remove extra subplot
    if len(categorical_features) < len(axes):
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig('docs/categorical_features_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Analyze each categorical feature
    for feature in categorical_features:
        logger.info(f"\n{feature.upper()}:")
        feature_counts = df[feature].value_counts()
        logger.info(f"Categories: {feature_counts.to_dict()}")
        
        # Target distribution for each category
        for category in df[feature].unique():
            category_data = df[df[feature] == category]
            disease_rate = category_data['target'].mean() * 100
            logger.info(f"  {category}: {len(category_data)} samples, {disease_rate:.1f}% have heart disease")

def create_correlation_heatmap(df):
    """Create correlation heatmap for numerical features"""
    numerical_features = ['age', 'resting_blood_pressure', 'cholesterol', 
                         'max_heart_rate_achieved', 'st_depression', 'num_major_vessels', 'target']
    
    # Calculate correlation matrix
    correlation_matrix = df[numerical_features].corr()
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    plt.title('Correlation Heatmap of Numerical Features')
    plt.tight_layout()
    plt.savefig('docs/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Correlation Heatmap created and saved")
    
    # Highlight important correlations
    target_correlations = correlation_matrix['target'].sort_values(key=abs, ascending=False)
    logger.info(f"\nTop correlations with target:")
    for feature, corr in target_correlations.items():
        if feature != 'target':
            logger.info(f"  {feature}: {corr:.3f}")
    
    return correlation_matrix

def create_pair_plots(df):
    """Create pair plots for key numerical features"""
    key_features = ['age', 'max_heart_rate_achieved', 'st_depression', 'num_major_vessels', 'target']
    
    # Create pair plot
    pair_data = df[key_features].copy()
    pair_data['target'] = pair_data['target'].map({0: 'No Disease', 1: 'Heart Disease'})
    
    # Use seaborn pairplot
    pair_plot = sns.pairplot(pair_data, hue='target', diag_kind='hist', 
                            plot_kws={'alpha': 0.6}, height=2)
    pair_plot.fig.suptitle('Pair Plot of Key Features', y=1.02)
    pair_plot.fig.set_size_inches(12, 10)
    
    plt.savefig('docs/pair_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Pair plot created and saved")

def create_feature_importance_plot(correlations):
    """Create feature importance plot based on correlations"""
    # Sort correlations by absolute value
    sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    features = [item[0] for item in sorted_correlations]
    corr_values = [item[1] for item in sorted_correlations]
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.barh(features, corr_values, color=['red' if x < 0 else 'blue' for x in corr_values])
    plt.xlabel('Correlation with Target')
    plt.title('Feature Importance Based on Correlation with Target')
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, corr_values)):
        plt.text(value + (0.01 if value >= 0 else -0.01), bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}', ha='left' if value >= 0 else 'right', va='center')
    
    plt.tight_layout()
    plt.savefig('docs/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Feature importance plot created and saved")

def generate_eda_report(df):
    """Generate comprehensive EDA report"""
    logger.info("Starting Exploratory Data Analysis...")
    
    # Create docs directory if it doesn't exist
    os.makedirs('docs', exist_ok=True)
    
    # 1. Target distribution analysis
    target_counts, target_percentages = analyze_target_distribution(df)
    
    # 2. Numerical features analysis
    stats_summary, correlations = analyze_numerical_features(df)
    
    # 3. Categorical features analysis
    analyze_categorical_features(df)
    
    # 4. Correlation analysis
    correlation_matrix = create_correlation_heatmap(df)
    
    # 5. Pair plots
    create_pair_plots(df)
    
    # 6. Feature importance
    create_feature_importance_plot(correlations)
    
    logger.info("EDA completed successfully! All visualizations saved to docs/ directory")
    
    return {
        'target_distribution': target_counts,
        'statistical_summary': stats_summary,
        'correlations': correlations,
        'correlation_matrix': correlation_matrix
    }

def main():
    """Main function to run the EDA pipeline"""
    try:
        # Load processed data
        df = load_processed_data()
        
        # Generate EDA report
        eda_results = generate_eda_report(df)
        
        logger.info("Exploratory Data Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"EDA failed: {e}")
        raise

if __name__ == "__main__":
    main()
