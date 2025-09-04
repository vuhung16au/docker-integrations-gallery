# Use official Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user
RUN useradd -m -s /bin/bash jupyter && \
    chown -R jupyter:jupyter /app

# Create Jupyter config directory
RUN mkdir -p /home/jupyter/.jupyter && \
    chown -R jupyter:jupyter /home/jupyter/.jupyter

# Copy Jupyter configuration
COPY jupyter_server_config.py /home/jupyter/.jupyter/

# Switch to non-root user
USER jupyter

# Expose port for Jupyter
EXPOSE 8888

# Run Jupyter Lab without authentication
CMD ["jupyter", "lab", "--config=/home/jupyter/.jupyter/jupyter_server_config.py"]