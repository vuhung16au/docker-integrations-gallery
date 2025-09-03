# Multi-Stage Docker Build Project

This project demonstrates Docker multi-stage builds for machine learning model training and serving. It shows how to create efficient, production-ready Docker images by separating the training and serving environments.

## 🎯 Project Overview

The project implements a complete ML pipeline using Docker multi-stage builds:

1. **Base Stage**: Common dependencies (pandas, numpy)
2. **Training Stage**: Model training with scikit-learn
3. **Serving Stage**: Minimal production image with Flask server

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Base Stage    │    │ Training Stage  │    │ Serving Stage   │
│                 │    │                 │    │                 │
│ • Python 3.9    │───▶│ • scikit-learn  │───▶│ • Flask         │
│ • pandas        │    │ • train_model.py│    │ • serve_model.py│
│ • numpy         │    │ • model.pkl     │    │ • model.pkl     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
01.Multi-Stage-Build/
├── Dockerfile          # Multi-stage build configuration
├── requirements.txt    # Python dependencies
├── train_model.py     # Model training script
└── README.md          # This file
```

## 🚀 Quick Start

### 1. Build the Docker Image

```bash
cd 01.Multi-Stage-Build
docker build -t ml-multistage .
```

### 2. Run the Container

```bash
docker run -p 5000:5000 ml-multistage
```

### 3. Test the API

```bash
# Health check
curl http://localhost:5000/health

# Get API info
curl http://localhost:5000/

# Make predictions
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"values": [1, 2, 3, 4, 5]}'
```

## 🔧 How It Works

### Multi-Stage Build Process

1. **Base Stage**: Installs common dependencies used across stages
2. **Training Stage**: 
   - Inherits from base stage
   - Installs scikit-learn
   - Runs training script to generate `model.pkl`
3. **Serving Stage**:
   - Starts fresh with minimal Python image
   - Installs only production dependencies (Flask, Gunicorn)
   - Copies trained model from training stage
   - Exposes port 5000 and runs the server

### Benefits of Multi-Stage Builds

- **Smaller Production Image**: Only serving dependencies included
- **Security**: Training dependencies not exposed in production
- **Efficiency**: Reuses layers across stages
- **Reproducibility**: Consistent training and serving environments

## 📊 Model Details

The project trains a simple linear regression model:
- **Equation**: y = 2x + 1 + noise
- **Training Data**: 100 synthetic data points
- **Features**: Single numerical input
- **Output**: Continuous numerical prediction

## 🧪 Testing the Model

### Example Predictions

```bash
# Test with single value
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"values": [5]}'

# Expected output: {"predictions": [11.0], ...}

# Test with multiple values
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"values": [0, 1, 2, 3, 4]}'

# Expected output: {"predictions": [1.0, 3.0, 5.0, 7.0, 9.0], ...}
```

## 🐳 Docker Commands Reference

### Build Commands

```bash
# Build with specific stage
docker build --target training -t ml-training .

# Build with specific stage
docker build --target serving -t ml-serving .

# Build all stages (default)
docker build -t ml-multistage .
```

### Run Commands

```bash
# Run with port mapping
docker run -p 5000:5000 ml-multistage

# Run in background
docker run -d -p 5000:5000 --name ml-server ml-multistage

# Run with custom environment
docker run -p 5000:5000 -e PORT=8080 ml-multistage
```
