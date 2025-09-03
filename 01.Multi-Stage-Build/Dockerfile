# Multi-stage build demo for ML model training and serving
# This demonstrates how to use Docker multi-stage builds to:
# 1. Train a model in one stage
# 2. Serve the model in a minimal production stage

# Base stage with common dependencies
FROM python:3.9-slim as base
WORKDIR /app
RUN pip install --no-cache-dir pandas==2.1.4 numpy==1.24.3

# Training stage - train the model
FROM base as training
RUN pip install --no-cache-dir scikit-learn==1.3.2
COPY train_model.py .
RUN python train_model.py

# Serving stage - minimal production image
FROM python:3.9-slim as serving
WORKDIR /app
RUN pip install --no-cache-dir flask==3.0.0 gunicorn==21.2.0 numpy==1.24.3

# Copy only the trained model and serving code from training stage
COPY --from=training /app/model.pkl .
COPY serve_model.py .

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run the server
CMD ["python", "serve_model.py"]