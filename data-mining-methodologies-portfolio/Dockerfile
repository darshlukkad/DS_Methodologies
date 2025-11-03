FROM python:3.9-slim

LABEL maintainer="Data Science Portfolio"
LABEL description="Data Mining Methodologies - CRISP-DM, SEMMA, KDD"

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Create necessary directories
RUN mkdir -p /root/.kaggle && \
    mkdir -p crisp_dm/data/{raw,processed} && \
    mkdir -p semma/data/{raw,processed} && \
    mkdir -p kdd/data/{raw,processed} && \
    mkdir -p crisp_dm/reports && \
    mkdir -p semma/reports && \
    mkdir -p kdd/reports

# Expose ports
# 8888 for Jupyter
# 8000 for FastAPI
EXPOSE 8888 8000

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV JUPYTER_ENABLE_LAB=yes

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8888/ || exit 1

# Default command: Start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
