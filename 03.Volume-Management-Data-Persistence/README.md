# Volume Management & Data Persistence Project

This project demonstrates Docker volume management, data persistence, and multi-service data processing. It shows how to create, manage, and persist data across container restarts using different types of volumes and backup strategies.

## 🎯 Project Overview

The project implements a comprehensive data processing pipeline with volume management:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Data Processor │    │  File Watcher   │    │ Backup Service  │    │ Log Aggregator  │
│                 │    │                 │    │                 │    │                 │
│ • Processes     │    │ • Monitors      │    │ • Creates       │    │ • Collects      │
│   CSV files     │    │   file changes  │    │   backups       │    │   logs          │
│ • Generates     │    │ • Logs events   │    │ • Manages       │    │ • Analyzes      │
│   summaries     │    │ • Triggers      │    │   retention     │    │   patterns      │
│ • Serves API    │    │   processing    │    │ • Restores      │    │ • Aggregates    │
│   endpoints     │    │                 │    │   data          │    │   statistics    │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │                       │
         └───────────────────────┼───────────────────────┼───────────────────────┘
                                 │                       │
                    ┌─────────────────┐    ┌─────────────────┐
                    │  Named Volumes  │    │  Bind Mounts    │
                    │                 │    │                 │
                    │ • input_data    │    │ • ./data/local  │
                    │ • output_data   │    │   (read-only)   │
                    │ • backup_data   │    │                 │
                    │ • logs          │    │                 │
                    └─────────────────┘    └─────────────────┘
```

## 🏗️ Architecture Components

### 1. **Data Processor Service** (`data-processor`)
- **Purpose**: Main data processing and API service
- **Features**: CSV processing, data transformation, REST API
- **Volumes**: All named volumes (input, output, backup, logs)
- **Port**: 8080 (exposed for API access)

### 2. **File Watcher Service** (`file-watcher`)
- **Purpose**: Monitor input directory for file changes
- **Features**: Real-time file system monitoring, event logging
- **Volumes**: input_data, logs
- **Dependencies**: data-processor

### 3. **Backup Service** (`backup-service`)
- **Purpose**: Automated backup and data retention
- **Features**: Scheduled backups, retention policies, restore functionality
- **Volumes**: output_data, backup_data, logs
- **Dependencies**: data-processor

### 4. **Log Aggregator Service** (`log-aggregator`)
- **Purpose**: Log collection and analysis
- **Features**: Log parsing, pattern analysis, temporary storage
- **Volumes**: logs, tmpfs mount
- **Dependencies**: All services

## 📁 Project Structure

```
03.Volume-Management-Data-Persistence/
├── Dockerfile                    # Multi-service container configuration
├── docker-compose.yaml          # Service orchestration with volumes
├── requirements.txt             # Python dependencies
├── data_processor.py            # Main data processing service
├── file_watcher.py              # File system monitoring service
├── backup_service.py            # Automated backup service
├── log_aggregator.py            # Log collection and analysis
├── scripts/                     # Utility scripts
│   ├── setup_data.sh           # Data setup script
│   └── test_volumes.sh         # Volume testing script
├── data/                        # Local data directory
│   └── local/                  # Sample data files (bind mounted)
│       └── README.md           # Data directory documentation
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Setup Sample Data

```bash
cd 03.Volume-Management-Data-Persistence
chmod +x scripts/setup_data.sh
./scripts/setup_data.sh
```

### 2. Build and Start Services

```bash
docker-compose up --build
```

### 3. Test Volume Functionality

```bash
chmod +x scripts/test_volumes.sh
./scripts/test_volumes.sh
```

### 4. Access the API

```bash
# Health check
curl http://localhost:8080/health

# Get service status
curl http://localhost:8080/status

# Generate sample data
curl -X POST http://localhost:8080/generate-data \
  -H "Content-Type: application/json" \
  -d '{"filename": "test_data.csv", "rows": 100}'

# Process data
curl -X POST http://localhost:8080/process \
  -H "Content-Type: application/json" \
  -d '{"filename": "test_data.csv"}'

# List files
curl http://localhost:8080/files

# Get volume information
curl http://localhost:8080/volumes
```

## 🔧 How It Works

### Volume Types Demonstrated

1. **Named Volumes**:
   - `input_data`: Persistent storage for input files
   - `output_data`: Persistent storage for processed files
   - `backup_data`: Persistent storage for backups
   - `logs`: Persistent storage for log files

2. **Bind Mounts**:
   - `./data/local:/app/data/local:ro`: Read-only access to local data

3. **tmpfs Mounts**:
   - `/tmp`: Temporary storage for log analysis

### Data Flow

1. **Input**: Files placed in `input_data` volume
2. **Processing**: Data processor transforms CSV files
3. **Output**: Processed files saved to `output_data` volume
4. **Backup**: Backup service creates periodic backups
5. **Monitoring**: File watcher monitors for changes
6. **Logging**: All services write to shared `logs` volume
7. **Analysis**: Log aggregator analyzes collected logs

### Persistence Features

- **Data Survival**: All data persists across container restarts
- **Backup Strategy**: Automated backups with retention policies
- **Volume Management**: Named volumes for reliable storage
- **Log Aggregation**: Centralized logging with analysis

## 📊 API Endpoints

### Data Processor Service (Port 8080)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/status` | GET | Processing status and statistics |
| `/generate-data` | POST | Generate sample data |
| `/process` | POST | Process files (specific or all) |
| `/files` | GET | List files in input/output directories |
| `/volumes` | GET | Volume information and disk usage |

### Example API Usage

```bash
# Generate and process data
curl -X POST http://localhost:8080/generate-data \
  -H "Content-Type: application/json" \
  -d '{"filename": "sample.csv", "rows": 50}'

curl -X POST http://localhost:8080/process \
  -H "Content-Type: application/json" \
  -d '{"filename": "sample.csv"}'

# Check results
curl http://localhost:8080/files
curl http://localhost:8080/status
```

## 🧪 Testing

### Automated Testing

Run the comprehensive test suite:

```bash
./scripts/test_volumes.sh
```

The test script verifies:
- ✅ Service health and connectivity
- ✅ Volume mounting and accessibility
- ✅ Data generation and processing
- ✅ File listing and management
- ✅ Data persistence across restarts
- ✅ Backup functionality
- ✅ Log aggregation

### Manual Testing

1. **Data Persistence Test**:
   ```bash
   # Generate data
   curl -X POST http://localhost:8080/generate-data -d '{"rows": 100}'
   
   # Restart services
   docker-compose restart
   
   # Verify data persisted
   curl http://localhost:8080/files
   ```

2. **Volume Inspection**:
   ```bash
   # List volumes
   docker volume ls
   
   # Inspect volume
   docker volume inspect volume-management-data-persistence_input_data
   
   # Check volume contents
   docker run --rm -v volume-management-data-persistence_input_data:/data alpine ls -la /data
   ```

## 🐳 Docker Commands Reference

### Volume Management

```bash
# List all volumes
docker volume ls

# Inspect specific volume
docker volume inspect volume-management-data-persistence_input_data

# Remove unused volumes
docker volume prune

# Backup volume data
docker run --rm -v volume-management-data-persistence_input_data:/data -v $(pwd):/backup alpine tar czf /backup/input_data_backup.tar.gz -C /data .

# Restore volume data
docker run --rm -v volume-management-data-persistence_input_data:/data -v $(pwd):/backup alpine tar xzf /backup/input_data_backup.tar.gz -C /data
```

### Service Management

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Restart specific service
docker-compose restart data-processor

# Scale services
docker-compose up -d --scale data-processor=2

# Stop and remove volumes
docker-compose down -v
```

### Debugging

```bash
# Access container shell
docker exec -it volume-data-processor /bin/bash

# Check volume mounts
docker exec volume-data-processor df -h

# View log files
docker exec volume-data-processor ls -la /app/logs/

# Monitor file system events
docker exec volume-file-watcher tail -f /app/logs/file_watcher.log
```

## 🔍 Volume Types Explained

### 1. Named Volumes
- **Purpose**: Persistent data storage managed by Docker
- **Location**: Docker's internal storage
- **Benefits**: Portable, managed by Docker, survives container removal
- **Use Case**: Database data, application state, logs

### 2. Bind Mounts
- **Purpose**: Direct access to host filesystem
- **Location**: Host filesystem paths
- **Benefits**: Direct access, easy debugging, development workflow
- **Use Case**: Configuration files, source code, read-only data

### 3. tmpfs Mounts
- **Purpose**: In-memory temporary storage
- **Location**: Container's memory
- **Benefits**: Fast access, automatic cleanup
- **Use Case**: Temporary files, cache, sensitive data

## 🛠️ Customization

### Environment Variables

```bash
# Data processing
PROCESSING_MODE=continuous
LOG_LEVEL=INFO

# Backup settings
BACKUP_INTERVAL=300
RETENTION_DAYS=7

# File watching
WATCH_DIR=/app/data/input

# Log aggregation
AGGREGATION_INTERVAL=60
```

### Volume Configuration

Modify `docker-compose.yaml` to customize volumes:

```yaml
volumes:
  input_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /path/to/host/input
```

## 🚨 Troubleshooting

### Common Issues

1. **Permission Denied**:
   ```bash
   # Fix permissions
   sudo chown -R $USER:$USER data/
   chmod -R 755 data/
   ```

2. **Volume Not Mounted**:
   ```bash
   # Check volume exists
   docker volume ls | grep volume-management
   
   # Recreate volumes
   docker-compose down -v
   docker-compose up -d
   ```

3. **Services Not Starting**:
   ```bash
   # Check logs
   docker-compose logs data-processor
   
   # Check port conflicts
   netstat -tulpn | grep 8080
   ```

4. **Data Not Persisting**:
   ```bash
   # Verify volume mounts
   docker exec volume-data-processor mount | grep data
   
   # Check volume contents
   docker run --rm -v volume-management-data-persistence_output_data:/data alpine ls -la /data
   ```

### Debugging Commands

```bash
# Service status
docker-compose ps

# Resource usage
docker stats

# Volume usage
docker system df -v

# Container inspection
docker inspect volume-data-processor
```

## 📚 Learning Outcomes

After completing this project, you will understand:

- ✅ **Volume Types**: Named volumes, bind mounts, tmpfs mounts
- ✅ **Data Persistence**: How to ensure data survives container restarts
- ✅ **Volume Management**: Creating, inspecting, and managing volumes
- ✅ **Multi-Service Architecture**: Coordinating services with shared volumes
- ✅ **Backup Strategies**: Automated backup and retention policies
- ✅ **Log Management**: Centralized logging and analysis
- ✅ **File System Monitoring**: Real-time file change detection
- ✅ **API Development**: RESTful services with volume integration

---

**Happy Learning! 🐳**

This project demonstrates essential volume management concepts that are crucial for production Docker deployments. Master these concepts to build robust, data-persistent containerized applications.
