# Multi-Container Data Pipeline Dockerfile
# This image will be used for data preparation, model training, and model serving

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files in pipeline sequence order
COPY 00-prepare_data.py .
COPY 10-train_model.py .
COPY 90-serve_model.py .

# Create a non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose port for the Flask server
EXPOSE 5000

# Default command (can be overridden in docker-compose)
CMD ["python", "90-serve_model.py"]
