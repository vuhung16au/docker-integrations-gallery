# Jupyter Lab Docker Project

A containerized Jupyter Lab environment with comprehensive data science and machine learning packages, designed for interactive development and analysis.

## Project Structure

```
.
├── Dockerfile                    # Docker image configuration
├── requirements.txt              # Python dependencies
├── jupyter_server_config.py     # Jupyter authentication configuration
├── test-container.sh            # Container testing utility
└── README.md                    # This file
```

## Features

- **Jupyter Lab 4.0+**: Modern web-based interactive development environment
- **Data Science Stack**: NumPy, Pandas, Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn, SciPy
- **Advanced Visualization**: Plotly, Bokeh
- **Interactive Widgets**: IPyWidgets for enhanced notebooks
- **Security**: Runs as non-root user
- **Optimized**: Multi-layer caching and minimal image size

## Prerequisites

- Docker installed on your system
- Git (optional, for version control)

## Quick Start

### 1. Build the Docker Image

```bash
cd 06.Jupyter
docker build -t jupyter-lab .
```

### 2. Run the Container

```bash
docker run -d \
  --name jupyter-container \
  -p 8888:8888 \
  -v $(pwd):/app/workspace \
  jupyter-lab
```

### 3. Access Jupyter Lab

Open your browser and navigate to:
```
http://localhost:8888/lab
```

**No authentication required!** The container is configured to run without tokens or passwords for easy access.

## Project Details

### Dockerfile

The Dockerfile uses Python 3.11 slim image and:
- Sets up a proper working directory (`/app`)
- Installs system dependencies for scientific computing
- Installs Python packages from `requirements.txt`
- Creates a non-root user for security
- Exposes port 8888 for Jupyter Lab
- Runs Jupyter Lab with authentication disabled for easy access

### Configuration

The `jupyter_server_config.py` file configures Jupyter to:
- Disable token authentication (`c.ServerApp.token = ''`)
- Disable password authentication (`c.ServerApp.password = ''`)
- Allow remote access (`c.ServerApp.allow_remote_access = True`)
- Allow all origins (`c.ServerApp.allow_origin = '*'`)

### Dependencies

#### Core Jupyter Packages
- **jupyter**: Core Jupyter functionality
- **jupyterlab**: Modern web-based interface
- **notebook**: Classic notebook interface

#### Data Science Packages
- **numpy**: Numerical computing
- **pandas**: Data manipulation and analysis
- **matplotlib**: Basic plotting
- **seaborn**: Statistical data visualization

#### Machine Learning Packages
- **scikit-learn**: Machine learning algorithms
- **scipy**: Scientific computing

#### Visualization Packages
- **plotly**: Interactive plots
- **bokeh**: Interactive web-based visualization

#### Additional Packages
- **ipywidgets**: Interactive widgets
- **tqdm**: Progress bars
- **requests**: HTTP library

## Docker Commands Reference

### Build Commands
```bash
# Build with custom tag
docker build -t my-jupyter:latest .

# Build without cache
docker build --no-cache -t jupyter-lab .

# Build with specific platform
docker build --platform linux/amd64 -t jupyter-lab .
```

### Run Commands
```bash
# Basic run
docker run -d -p 8888:8888 jupyter-lab

# Run with volume mounting for persistent data
docker run -d \
  --name jupyter-persistent \
  -p 8888:8888 \
  -v $(pwd)/data:/app/workspace/data \
  jupyter-lab

# Run with custom port
docker run -d -p 9000:8888 jupyter-lab

# Run with environment variables
docker run -d \
  -p 8888:8888 \
  -e JUPYTER_ENABLE_LAB=yes \
  jupyter-lab
```

### Management Commands
```bash
# List running containers
docker ps

# View container logs
docker logs jupyter-container

# Stop the container
docker stop jupyter-container

# Remove the container
docker rm jupyter-container

# List images
docker images

# Remove the image
docker rmi jupyter-lab
```

## Advanced Usage

### Volume Mounting

Mount your local directories to persist data and notebooks:

```bash
docker run -d \
  --name jupyter-workspace \
  -p 8888:8888 \
  -v $(pwd)/notebooks:/app/workspace/notebooks \
  -v $(pwd)/data:/app/workspace/data \
  -v $(pwd)/models:/app/workspace/models \
  jupyter-lab
```

### Custom Jupyter Configuration

Create a custom `jupyter_config.py` file and mount it:

```bash
docker run -d \
  -p 8888:8888 \
  -v $(pwd)/jupyter_config.py:/app/.jupyter/jupyter_server_config.py \
  jupyter-lab
```

### Running with Docker Compose

Create a `docker-compose.yml` file:

```yaml
version: '3.8'
services:
  jupyter:
    build: .
    ports:
      - "8888:8888"
    volumes:
      - ./workspace:/app/workspace
    environment:
      - JUPYTER_ENABLE_LAB=yes
    restart: unless-stopped
```

Then run:
```bash
docker-compose up -d
```

## Troubleshooting

### Common Issues

1. **Port already in use**: Check if port 8888 is available with `lsof -i :8888`
2. **Permission denied**: Ensure proper volume permissions
3. **Container won't start**: Check logs with `docker logs jupyter-container`
4. **Can't access Jupyter**: Verify the token from container logs

### Debugging

```bash
# Run container interactively
docker run -it jupyter-lab /bin/bash

# View container logs
docker logs jupyter-container

# Inspect container
docker inspect jupyter-container

# Execute commands in running container
docker exec -it jupyter-container /bin/bash
```

### Performance Optimization

```bash
# Run with resource limits
docker run -d \
  --name jupyter-optimized \
  -p 8888:8888 \
  --memory=4g \
  --cpus=2 \
  jupyter-lab

# Use host networking (Linux only)
docker run -d \
  --name jupyter-host \
  --network host \
  jupyter-lab
```

