# Volume Management & Data Persistence Project
# This Dockerfile demonstrates volume management and data persistence concepts

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY *.py ./
COPY scripts/ ./scripts/

# Create directories for data persistence
RUN mkdir -p /app/data/input /app/data/output /app/data/backup /app/logs

# Set permissions for data directories
RUN chmod -R 755 /app/data /app/logs

# Create a non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose port for the web interface
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command - can be overridden in docker-compose
CMD ["python", "data_processor.py"]
