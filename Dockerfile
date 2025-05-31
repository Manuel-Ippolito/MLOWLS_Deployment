# Dockerfile
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies AND build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements_fastapi.txt .
COPY requirements_model.txt .
RUN pip install --no-cache-dir -r requirements_fastapi.txt
RUN pip install --no-cache-dir -r requirements_model.txt

# Copy application code
COPY ml_owls/ ./ml_owls/
COPY configs/ ./configs/
COPY models/ ./models/

# Create a non-root user and switch to it
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Add health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

# Expose the port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "ml_owls.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]