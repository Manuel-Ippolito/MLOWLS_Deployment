# Dockerfile
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
    LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/app \
    LABELSTUDIO_TOKEN="insert_your_token_here" 

# Install system dependencies
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      supervisor \
      git \
      build-essential \
      ffmpeg \
      libsndfile1 \
      curl \
      openssh-client \
 && rm -rf /var/lib/apt/lists/*

# Create user and group early
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory and change ownership to appuser
WORKDIR /app
RUN chown appuser:appuser /app

# Clone repository as appuser
USER appuser
RUN git clone --depth=1 \
        --branch main \
        https://github.com/Manuel-Ippolito/MLOWLS_Deployment.git /app

# Switch back to root to create system directories
USER root

# Create directories that need root permissions and set ownership
RUN mkdir -p /app/data /app/mlflow/mlruns /opt/venv /home/appuser/.ssh \
 && chown -R appuser:appuser /app/data /app/mlflow /opt/venv /home/appuser

# Switch back to appuser for remaining operations
USER appuser

# Copy SSH key
COPY --chown=appuser:appuser MLOPS_key.pem /home/appuser/.ssh/MLOPS_key.pem
RUN chmod 600 /home/appuser/.ssh/MLOPS_key.pem

# Create MLflow database file
RUN touch /app/mlflow/mlflow.db \
 && chown -R appuser:appuser /app/mlflow

# Install Python packages
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip setuptools wheel \
 && /opt/venv/bin/pip install --no-cache-dir -e ".[inference]"

# Configure DVC and pull initial data
RUN ssh-keyscan -H 57.151.108.196 >> /home/appuser/.ssh/known_hosts \
 && mkdir -p /home/appuser/.config/dvc \
 && chown -R appuser:appuser /home/appuser/.config \
 && cd /app \
 && /opt/venv/bin/dvc remote add -d -f MLOPSDocker ssh://MLOPSuser@57.151.108.196:/home/MLOPSuser/dvc-remote \
 && /opt/venv/bin/dvc remote modify MLOPSDocker --local keyfile /home/appuser/.ssh/MLOPS_key.pem \
 && /opt/venv/bin/dvc pull --verbose || echo "DVC pull failed"

# Switch back to root for supervisor config
USER root

# Create log directories and set proper permissions
RUN mkdir -p /var/log/supervisor \
 && touch /var/log/supervisor/supervisord.log \
          /var/log/labelstudio.log /var/log/labelstudio.err \
          /var/log/mlflow.log /var/log/mlflow.err \
          /var/log/api.log /var/log/api.err \
          /var/log/dvc-pull.log /var/log/dvc-pull.err \
          /var/log/dvc-synch.log /var/log/dvc-sync.err \
 && chown root:root /var/log/supervisor/supervisord.log \
 && chmod 644 /var/log/supervisor/supervisord.log \
 && chown appuser:appuser /var/log/labelstudio.* /var/log/mlflow.* /var/log/api.* /var/log/dvc-*

# Copy supervisord config as main config file
COPY supervisord.conf /etc/supervisor/supervisord.conf

EXPOSE 8080 5000 8000

# Run supervisor as root
CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/supervisord.conf"]