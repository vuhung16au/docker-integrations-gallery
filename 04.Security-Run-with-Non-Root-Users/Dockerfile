FROM python:3.11-slim

# Create a non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application files
COPY . .

# Change ownership of the app directory to the non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port (if needed for future use)
EXPOSE 8000

# Run the application
CMD ["python", "hello_non_root.py"]