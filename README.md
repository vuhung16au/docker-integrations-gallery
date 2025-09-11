# Docker Integrations Gallery

A comprehensive collection of Docker projects designed to teach containerization, orchestration, and modern DevOps practices through hands-on examples. This gallery provides progressive learning experiences from basic Docker concepts to advanced multi-container data science pipelines.

Notes: This is a learning resource for Docker integration with Machine Learning, not learning Machine Learning.

## 🎯 Project Overview

The Docker Integrations Gallery is a structured learning resource that demonstrates real-world applications of Docker technology across different domains. Each project builds upon previous concepts, creating a comprehensive learning path that covers:

- **Containerization Fundamentals**: Basic Docker concepts and best practices
- **Multi-Stage Builds**: Optimizing Docker images for production environments
- **Multi-Container Orchestration**: Managing complex applications with Docker Compose
- **Security Best Practices**: Running containers with non-root users and security hardening
- **Data Science Pipelines**: End-to-end ML workflows with containerized services
- **Development Environments**: Jupyter notebooks and interactive development setups

The gallery emphasizes practical, production-ready implementations that developers can adapt for their own projects.

## 👥 Target Audience

This project is designed for:

### **Beginners to Docker**
- Developers new to containerization
- Students learning DevOps practices
- Professionals transitioning to containerized workflows

### **Data Scientists & ML Engineers**
- Practitioners building ML pipelines
- Teams implementing MLOps practices
- Researchers needing reproducible environments

### **DevOps Engineers**
- Professionals optimizing deployment processes
- Teams implementing CI/CD pipelines
- Engineers managing microservices architectures

### **Software Developers**
- Full-stack developers adopting containerization
- Backend developers building scalable applications
- Frontend developers working with containerized environments

## 🎓 Learning Outcomes

Upon completing this gallery, learners will be able to:

### **Docker Fundamentals**
- ✅ Create and manage Docker containers
- ✅ Build optimized Docker images
- ✅ Understand Docker layers and caching
- ✅ Implement multi-stage builds for production

### **Container Orchestration**
- ✅ Design multi-container applications
- ✅ Use Docker Compose for service orchestration
- ✅ Manage data persistence with volumes
- ✅ Configure container networking

### **Security & Best Practices**
- ✅ Implement security best practices in containers
- ✅ Run containers with non-root users
- ✅ Optimize image sizes and security
- ✅ Follow Docker security guidelines

### **Data Science & ML Pipelines**
- ✅ Containerize machine learning workflows
- ✅ Build end-to-end data science pipelines
- ✅ Implement model serving with APIs
- ✅ Create reproducible ML environments

### **Production Deployment**
- ✅ Deploy containerized applications
- ✅ Monitor and debug containerized services
- ✅ Implement health checks and logging
- ✅ Scale containerized applications

## 🛠️ Technical Stack

### **Core Technologies**
- **Docker**: Containerization platform
- **Docker Compose**: Multi-container orchestration
- **Python 3.9+**: Primary programming language
- **Linux**: Container runtime environment

### **Data Science & ML**
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **CatBoost**: Gradient boosting framework
- **LightGBM**: Gradient boosting framework
- **XGBoost**: Gradient boosting framework
- **Flask**: Web framework for API development
- **Jupyter**: Interactive development environment

### **Databases & Storage**
- **PostgreSQL**: Relational database
- **Docker Volumes**: Persistent data storage
- **CSV**: Data file formats

### **Development Tools**
- **Git**: Version control
- **curl**: API testing
- **bash**: Shell scripting
- **Docker CLI**: Container management

### **Infrastructure**
- **Multi-stage builds**: Optimized image creation
- **Health checks**: Service monitoring
- **Port mapping**: Network configuration
- **Environment variables**: Configuration management

## 📁 Project Structure & Learning Path

The gallery is organized as a progressive learning journey, with each project building upon previous concepts:

### **Phase 1: Docker Fundamentals**
```
00.HelloDocker/
├── Dockerfile          # Basic containerization
├── hello.py           # Simple Python application
├── requirements.txt    # Dependencies management
└── README.md          # Project documentation
```
**Learning Focus**: Basic Docker concepts, image building, container execution

### **Phase 2: Advanced Build Techniques**
```
01.Multi-Stage-Build/
├── Dockerfile          # Multi-stage build configuration
├── train_model.py     # Model training script
├── serve_model.py     # Model serving application
├── requirements.txt    # Dependencies
└── README.md          # Project documentation
```
**Learning Focus**: Multi-stage builds, production optimization, ML model serving

### **Phase 3: Multi-Container Orchestration**
```
02.Multi-Container-Data-Pipelines/
├── docker-compose.yaml # Service orchestration
├── Dockerfile          # Application container
├── 00-prepare_data.py  # Data preparation service
├── 10-train_model.py   # Model training service
├── 90-serve_model.py   # Model serving service
├── requirements.txt    # Dependencies
└── README.md          # Project documentation
```
**Learning Focus**: Service orchestration, data flow, inter-service communication

### **Phase 4: Volume Management & Data Persistence**
```
03.Volume-Management-Data-Persistence/
├── docker-compose.yaml # Multi-service setup with volumes
├── Dockerfile          # Application container
├── data_processor.py   # Main data processing service
├── file_watcher.py     # File monitoring service
├── backup_service.py   # Automated backup service
├── log_aggregator.py   # Log collection and analysis
├── scripts/            # Utility scripts
│   ├── setup_data.sh  # Data setup script
│   └── test_volumes.sh # Volume testing script
├── data/local/         # Sample data (bind mounted)
├── requirements.txt    # Dependencies
└── README.md          # Project documentation
```
**Learning Focus**: Volume types, data persistence, backup strategies, log management

### **Phase 5: Security Best Practices**
```
04.Security-Run-with-Non-Root-Users/
├── Dockerfile          # Security-hardened container
├── hello_non_root.py   # Application with security focus
├── requirements.txt    # Dependencies
└── README.md          # Project documentation
```
**Learning Focus**: Container security, non-root execution, security hardening

### **Phase 6: Real-World Data Science Pipeline**
```
05.Heart-Disease-Predictions/
├── docker-compose.yaml # Multi-service orchestration
├── Dockerfile          # Application container
├── Dataset-Heart/      # Data storage
│   └── heart.csv      # Heart disease dataset
├── docs/              # Documentation
│   └── heart.md       # Dataset documentation
├── models/            # Trained models storage
├── 00-prepare_data.py # Data preparation
├── 20-exploratory_analysis.py # EDA
├── 40-train_models.py # Model training
├── 60-evaluate_models.py # Model evaluation
├── 80-run_pipeline.py # Pipeline orchestration
├── app.py             # Flask API server
├── test-endpoints.sh  # API testing script
├── requirements.txt   # Dependencies
└── README.md         # Project documentation
```
**Learning Focus**: End-to-end ML pipeline, multiple ML algorithms, API development, comprehensive testing

### **Phase 7: Development Environment**
```
06.Jupyter/
├── Dockerfile              # Jupyter environment
├── jupyter_server_config.py # Jupyter configuration
├── test-container.sh       # Container testing
├── requirements.txt        # Dependencies
└── README.md              # Project documentation
```
**Learning Focus**: Interactive development, Jupyter configuration, development workflows

## 🚀 Getting Started

### Prerequisites
- Docker installed on your system
- Git for version control
- Basic knowledge of Python (recommended)
- Command line familiarity

### Quick Start
1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd docker-integrations-gallery
   ```

2. **Start with Hello Docker**:
   ```bash
   cd 00.HelloDocker
   docker build -t hello-docker .
   docker run hello-docker
   ```

3. **Follow the learning path**: Progress through each numbered project in sequence

### Learning Recommendations
- **Complete projects in order**: Each builds upon previous concepts
- **Read all README files**: Comprehensive documentation for each project
- **Experiment with modifications**: Try changing configurations and see the results
- **Test thoroughly**: Use provided test scripts and create your own
- **Document your learnings**: Keep notes on concepts and best practices

## 🧪 Testing & Validation

### Automated Testing Script

The gallery includes a comprehensive test script that validates all Docker configurations to ensure they build and run correctly.

#### **Test Script: `scripts/test-docker.sh`**

This automated testing script provides:

- **Comprehensive Coverage**: Tests all 7 Docker configurations in the gallery
- **Build Validation**: Ensures all Dockerfiles and docker-compose.yaml files build successfully
- **Runtime Testing**: Verifies containers start and run properly
- **Smart Detection**: Handles both long-running services and one-time execution containers
- **Automatic Cleanup**: Removes containers, images, and volumes after testing
- **Clear Reporting**: Provides colored output with detailed success/failure status

#### **Usage**

Run the test script from the project root:

```bash
cd docker-integrations-gallery
./scripts/test-docker.sh
```

Note: The script takes a while to run, so get a cup of coffee.

#### **What Gets Tested**

The script automatically tests:

1. **00.HelloDocker** - Basic Dockerfile (simple hello world application)
2. **01.Multi-Stage-Build** - Advanced Dockerfile (ML model training/serving)
3. **02.Multi-Container-Data-Pipelines** - docker-compose.yaml (PostgreSQL + ML pipeline)
4. **03.Volume-Management-Data-Persistence** - docker-compose.yaml (volume management)
5. **04.Security-Run-with-Non-Root-Users** - Dockerfile (security best practices)
6. **05.Heart-Disease-Predictions** - docker-compose.yaml (Flask app with volumes)
7. **06.Jupyter** - Dockerfile (Jupyter Lab server)

#### **Test Output Example**

```
[INFO] Starting Docker Integration Gallery Tests...
[INFO] Docker binary: /usr/local/bin/docker
[INFO] ==========================================
[INFO] Testing: 00.HelloDocker (dockerfile)
[INFO] ==========================================
[INFO] Testing Dockerfile in 00.HelloDocker...
[INFO] Building image for hello-docker...
[INFO] Running container for hello-docker...
[SUCCESS] hello-docker container completed successfully
[SUCCESS] hello-docker test completed successfully
...
[INFO] ==========================================
[INFO] TEST SUMMARY
[INFO] ==========================================
[INFO] Total tests: 7
[SUCCESS] Passed: 7
[SUCCESS] Failed: 0
[SUCCESS] All tests passed! 🎉
```

#### **Prerequisites for Testing**

- Docker installed and running
- Sufficient disk space for building images
- Network access for pulling base images
- No conflicting services running on required ports

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests to improve this learning resource.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Happy Learning! 🐳**

Start your Docker journey with the Hello Docker project and work your way through the comprehensive learning path. Each project is designed to be hands-on, practical, and immediately applicable to real-world scenarios.