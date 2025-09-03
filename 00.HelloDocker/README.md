# Hello Docker Project

A simple Python application containerized with Docker to demonstrate basic Docker concepts.

## Project Structure

```
.
├── Dockerfile          # Docker image configuration
├── hello.py           # Main Python application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Prerequisites

- Docker installed on your system
- Git (optional, for version control)

## Quick Start

### 1. Build the Docker Image

```bash
docker build -t hello-docker .
```

### 2. Run the Container

```bash
docker run hello-docker
```

Expected output:
```
Hello, World!
```

## Project Details

### Dockerfile

The Dockerfile uses Python 3.11 slim image as the base and:
- Sets the working directory to `/app`
- Installs Python dependencies from `requirements.txt`
- Copies application files
- Runs the `hello.py` script

### Application

The `hello.py` file contains a simple Python script that prints "Hello, World!" to demonstrate basic Docker containerization.

### Dependencies

- **pandas**: Data manipulation library (included for demonstration purposes)

## Docker Commands Reference

### Build Commands
```bash
# Build with custom tag
docker build -t my-app:latest .

# Build without cache
docker build --no-cache -t hello-docker .
```

### Run Commands
```bash
# Run in foreground
docker run hello-docker

# Run in detached mode
docker run -d hello-docker

# Run with custom name
docker run --name my-container hello-docker
```

### Management Commands
```bash
# List running containers
docker ps

# List all containers
docker ps -a

# Stop a container
docker stop <container_id>

# Remove a container
docker rm <container_id>

# List images
docker images

# Remove an image
docker rmi hello-docker
```

## Troubleshooting

### Common Issues

1. **Port already in use**: Use `docker ps` to check for running containers
2. **Permission denied**: Ensure Docker daemon is running and you have proper permissions
3. **Image not found**: Verify the image was built successfully with `docker images`

### Debugging

```bash
# Run container interactively
docker run -it hello-docker /bin/bash

# View container logs
docker logs <container_id>

# Inspect container
docker inspect <container_id>
```

## Next Steps

This project demonstrates basic Docker concepts. Consider exploring:
- Multi-stage builds
- Docker Compose for multi-container applications
- Environment variables and configuration
- Volume mounting for persistent data
- Network configuration for container communication
