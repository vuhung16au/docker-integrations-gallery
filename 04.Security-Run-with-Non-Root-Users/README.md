# Docker Security: Running with Non-Root User

A secure Python application containerized with Docker that demonstrates security best practices by running the container with a non-root user.

## Project Structure

```
.
├── Dockerfile              # Secure Docker image configuration
├── hello_non_root.py       # Main Python application
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Security Features

This project demonstrates Docker security best practices:

- **Non-root user**: Container runs as `appuser` instead of root
- **Principle of least privilege**: Minimal permissions for the application
- **Secure file ownership**: Application files owned by the non-root user
- **No unnecessary capabilities**: Container runs with minimal privileges

## Prerequisites

- Docker installed on your system
- Git (optional, for version control)

## Quick Start

### 1. Build the Docker Image

```bash
docker build -t hello-secure .
```

### 2. Run the Container

```bash
docker run hello-secure
```

Expected output:
```
Hello, World! (non-root user)
```

### 3. Verify Security

Check that the container is running as a non-root user:

```bash
# Run container interactively to check user
docker run -it hello-secure /bin/bash
# Inside container, run: whoami
# Should show: appuser
```

## Project Details

### Dockerfile Security Features

The Dockerfile implements several security best practices:

1. **User Creation**: Creates a dedicated `appuser` group and user
2. **File Ownership**: Changes ownership of application files to the non-root user
3. **User Switching**: Uses `USER` directive to run as non-root
4. **Minimal Base Image**: Uses Python slim image for reduced attack surface
5. **Proper Permissions**: Sets appropriate file permissions

### Application

The `hello_non_root.py` file contains a simple Python script that prints "Hello, World! (non-root user)" to demonstrate secure Docker containerization.

### Dependencies

- **pandas**: Data manipulation library (included for demonstration purposes)

## Security Benefits

### Why Run as Non-Root?

1. **Reduced Attack Surface**: Limits potential damage from security vulnerabilities
2. **Container Escape Prevention**: Makes it harder for malicious processes to escape the container
3. **Host System Protection**: Prevents container processes from accessing host resources
4. **Compliance**: Meets security requirements for production deployments
5. **Best Practice**: Follows Docker security recommendations

### Security Considerations

- Container runs with minimal privileges
- No unnecessary packages or services
- Proper file permissions and ownership
- Non-root user cannot modify system files

## Docker Commands Reference

### Build Commands
```bash
# Build with custom tag
docker build -t my-secure-app:latest .

# Build without cache
docker build --no-cache -t hello-secure .
```

### Run Commands
```bash
# Run in foreground
docker run hello-secure

# Run in detached mode
docker run -d hello-secure

# Run with custom name
docker run --name my-secure-container hello-secure

# Run with security options
docker run --security-opt=no-new-privileges hello-secure
```

### Security Verification Commands
```bash
# Check container user
docker run --rm hello-secure whoami

# Inspect container security settings
docker inspect hello-secure | grep -A 10 "SecurityOpt"

# Check running processes
docker run --rm hello-secure ps aux
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
docker rmi hello-secure
```

## Troubleshooting

### Common Issues

1. **Permission denied**: Ensure Docker daemon is running and you have proper permissions
2. **User creation failed**: Check if the base image supports user creation
3. **File access issues**: Verify file ownership and permissions

### Debugging

```bash
# Run container interactively
docker run -it hello-secure /bin/bash

# View container logs
docker logs <container_id>

# Inspect container
docker inspect <container_id>

# Check user context
docker exec <container_id> whoami
```

## Security Best Practices

### Additional Security Measures

1. **Read-only root filesystem**:
   ```bash
   docker run --read-only hello-secure
   ```

2. **No new privileges**:
   ```bash
   docker run --security-opt=no-new-privileges hello-secure
   ```

3. **Resource limits**:
   ```bash
   docker run --memory=512m --cpus=1 hello-secure
   ```

4. **Network security**:
   ```bash
   docker run --network=none hello-secure
   ```

## Next Steps

This project demonstrates Docker security fundamentals. Consider exploring:

- **Multi-stage builds** for reduced attack surface
- **Secrets management** for sensitive data
- **Image scanning** for vulnerabilities
- **Runtime security** monitoring
- **Compliance frameworks** (CIS, NIST)
- **Container orchestration** security (Kubernetes, Docker Swarm)

## References

- [Docker Security Best Practices](https://docs.docker.com/engine/security/)
- [CIS Docker Benchmark](https://www.cisecurity.org/benchmark/docker/)
- [OWASP Container Security](https://owasp.org/www-project-container-security/)
