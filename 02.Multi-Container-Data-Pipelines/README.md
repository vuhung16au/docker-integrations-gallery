# Multi-Container Data Pipeline Project

This project demonstrates a complete data science pipeline using Docker Compose with multiple services working together. It shows how to design, implement, and orchestrate a production-ready ML pipeline with proper service separation and data flow management.

## 🎯 Project Overview

The project implements a complete ML pipeline with the following architecture:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │  Data Prep      │    │ Model Training  │    │ Model Serving   │
│   Database      │◄───┤  Service        │◄───┤ Service         │◄───┤ Service         │
│                 │    │                 │    │                 │    │                 │
│ • Stores raw    │    │ • Generates     │    │ • Reads data    │    │ • Serves API    │
│   and processed │    │   synthetic data│    │   from DB       │    │ • Loads trained │
│   data          │    │ • Creates       │    │ • Trains model  │    │   model         │
│ • Persistent    │    │   tables        │    │ • Saves model   │    │ • Handles       │
│   storage       │    │ • Stores data   │    │   to volume     │    │   predictions   │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🏗️ Architecture Components

### 1. **Database Service** (`database`)
- **Image**: PostgreSQL 13
- **Purpose**: Persistent data storage
- **Features**: Health checks, named volumes, network isolation
- **Port**: 5432 (exposed for debugging)

### 2. **Data Preparation Service** (`data-prep`)
- **Purpose**: Generate and store synthetic training data
- **Data**: Linear relationship y = 2x + 1 + noise
- **Output**: Creates `linear_data` table with 100 data points
- **Dependencies**: Waits for database to be healthy

### 3. **Model Training Service** (`model-train`)
- **Purpose**: Train linear regression model on stored data
- **Algorithm**: scikit-learn LinearRegression
- **Output**: Saves trained model as `model.pkl`
- **Dependencies**: Waits for data preparation to complete

### 4. **Model Serving Service** (`model-serve`)
- **Purpose**: Serve trained model via REST API
- **Framework**: Flask
- **Features**: Health checks, model loading, prediction endpoints
- **Port**: 5000 (exposed for external access)
- **Dependencies**: Waits for model training to complete

## 📁 Project Structure

```
02.Multi-Container-Data-Pipelines/
├── docker-compose.yaml    # Multi-service orchestration
├── Dockerfile             # Application container definition
├── requirements.txt       # Python dependencies
└── README.md             # This file

Pipeline files (numbered for sequence clarity):
├── 00-prepare_data.py    # Data generation and storage (Step 1)
├── 10-train_model.py     # Model training script (Step 2)
└── 90-serve_model.py     # Model serving API (Step 3)
```

## 🔢 **Pipeline File Naming Convention**

The pipeline files use a **numbered prefix system** to clearly indicate execution order and allow for easy insertion of new pipeline steps:

### **Current Pipeline Sequence**
- **`00-prepare_data.py`** - Data preparation and storage
- **`10-train_model.py`** - Model training and evaluation  
- **`90-serve_model.py`** - Model deployment and serving

### **Benefits of Numbered Naming**
1. **Clear Execution Order**: Numbers immediately show the pipeline sequence
2. **Easy Insertion**: Add new steps like `05-validate_data.py` or `15-feature_engineering.py`
3. **Logical Grouping**: 
   - `00-19`: Data preparation and validation
   - `20-39`: Feature engineering and preprocessing
   - `40-59`: Model training and evaluation
   - `60-79`: Model validation and testing
   - `80-99`: Deployment and serving
4. **Maintainability**: New team members can quickly understand the pipeline flow
5. **Scalability**: Easy to add intermediate steps without renumbering everything

### **Future Pipeline Expansion Examples**
```
00-prepare_data.py          # Data generation
05-validate_data.py         # Data quality checks
10-feature_engineering.py   # Feature creation
15-train_model.py           # Model training
20-evaluate_model.py        # Model evaluation
25-optimize_model.py        # Hyperparameter tuning
90-serve_model.py           # Model serving
95-monitor_model.py         # Model monitoring
```

## 🚀 Quick Start

### 1. **Start the Complete Pipeline**

```bash
cd 02.Multi-Container-Data-Pipelines
docker-compose up --build
```

This command will:
- Build the application image
- Start PostgreSQL database
- Run data preparation service (`00-prepare_data.py`)
- Run model training service (`10-train_model.py`)
- Start model serving service (`90-serve_model.py`)
- Wait for each service to complete before starting the next

To run the docker compose file

```bash
docker-compose up -d
```

### 2. **Monitor Pipeline Progress**

```bash
# View all services
docker-compose ps

# View logs for specific service
docker-compose logs database
docker-compose logs data-prep
docker-compose logs model-train
docker-compose logs model-serve

# View all logs
docker-compose logs -f
```

### 3. **Test the Pipeline**

```bash
# Health check
curl http://localhost:5000/health

# Get API information
curl http://localhost:5000/

# View sample data from database
curl http://localhost:5000/data/sample

# Make predictions
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"values": [5]}'
```

## 🔧 How It Works

### **Service Dependencies and Execution Order**

1. **Database Service** starts first and waits for health check
2. **Data Preparation** (`00-prepare_data.py`) starts after database is healthy
3. **Model Training** (`10-train_model.py`) starts after data preparation completes
4. **Model Serving** (`90-serve_model.py`) starts after model training completes

### **Data Flow**

1. **Data Generation**: `00-prepare_data.py` creates synthetic data (y = 2x + 1 + noise)
2. **Data Storage**: Data is stored in PostgreSQL `linear_data` table
3. **Model Training**: `10-train_model.py` reads data, trains model, saves to volume
4. **Model Serving**: `90-serve_model.py` loads model and serves predictions via API

### **Volume Management**

- **`postgres_data`**: Persistent PostgreSQL data storage
- **`model_data`**: Shared volume for trained models between services

## 🐳 Docker Compose Features

### **Health Checks**
- Database: Uses `pg_isready` to verify PostgreSQL availability
- Model Service: HTTP health check on `/health` endpoint

### **Service Dependencies**
- Uses `condition: service_healthy` for database dependency
- Uses `condition: service_completed_successfully` for pipeline stages

### **Networking**
- Custom `ml-network` bridge network for service communication
- Isolated service-to-service communication

### **Environment Variables**
- Database connection parameters
- Service-specific configurations
- Port mappings

## 📊 Data and Model Details

### **Training Data**
- **Equation**: y = 2x + 1 + noise
- **Range**: x ∈ [0, 20]
- **Samples**: 100 data points
- **Noise**: Normal distribution with σ = 0.5

### **Model**
- **Algorithm**: Linear Regression
- **Expected Coefficients**: slope ≈ 2.0, intercept ≈ 1.0
- **Evaluation**: Train/test split (80/20)
- **Metrics**: MSE, RMSE, R²

### **API Endpoints**
- `GET /`: API information and model status
- `GET /health`: Health check with database and model status
- `POST /predict`: Make predictions with input values
- `GET /data/sample`: View sample data from database

## 🧪 Testing the Pipeline

### **Complete Pipeline Test**

```bash
# Start the pipeline
docker-compose up --build

# Wait for all services to complete, then test
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"values": [5]}'

# Expected output: {"predictions": [[11.0]], ...}
```

### **Individual Service Testing**

```bash
# Test database connection
docker-compose exec database psql -U postgres -d dsproject -c "SELECT COUNT(*) FROM linear_data;"

# Test data preparation
docker-compose run --rm data-prep

# Test model training
docker-compose run --rm model-train

# Test model serving
docker-compose run --rm model-serve
```

## 🔍 Troubleshooting

### **Common Issues**

1. **Port Conflicts**
   ```bash
   # Check what's using port 5000
   lsof -i :5000
   
   # Modify docker-compose.yaml to use different port
   ports:
     - "8080:5000"
   ```

2. **Database Connection Issues**
   ```bash
   # Check database logs
   docker-compose logs database
   
   # Verify database is running
   docker-compose ps database
   ```

3. **Service Dependencies**
   ```bash
   # Check service status
   docker-compose ps
   
   # View dependency errors
   docker-compose logs
   ```

### **Debug Commands**

```bash
# Run service interactively
docker-compose run --rm data-prep bash

# Check service logs
docker-compose logs -f [service-name]

# Inspect volumes
docker volume ls
docker volume inspect [volume-name]

# Check network
docker network ls
docker network inspect [network-name]
```

## 📈 Performance Considerations

- **Parallel Execution**: Services run sequentially due to dependencies
- **Resource Management**: Each service has isolated resources
- **Volume Sharing**: Model data shared between training and serving
- **Health Checks**: Prevent service startup before dependencies are ready

## 🔒 Security Features

- **Non-root User**: Application runs as non-root user
- **Network Isolation**: Services communicate only through defined network
- **Volume Isolation**: Data persistence with controlled access
- **Environment Variables**: Sensitive data passed via environment

## 🚀 Production Deployment

### **Scaling Considerations**

```bash
# Scale specific services
docker-compose up --scale model-serve=3

# Use external database
docker-compose -f docker-compose.yaml -f docker-compose.prod.yaml up
```

### **Environment Configuration**

```bash
# Use environment file
docker-compose --env-file .env up

# Override specific variables
POSTGRES_PASSWORD=secure_password docker-compose up
```

### **Monitoring and Logging**

```bash
# Add logging driver
docker-compose up --log-driver=json-file

# Use external monitoring
docker-compose -f docker-compose.yaml -f docker-compose.monitoring.yaml up
```

## 📚 Learning Resources

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [PostgreSQL Docker Image](https://hub.docker.com/_/postgres)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [Multi-Service Architecture Patterns](https://microservices.io/)

