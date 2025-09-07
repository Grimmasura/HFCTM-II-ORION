# HFCTM-II-ORION Production Container
# Implements recursive stability and dependency isolation

FROM python:3.11.7-slim

# Metadata for HFCTM-II identification
LABEL maintainer="Joshua Robert Humphrey"
LABEL version="2.0"
LABEL description="HFCTM-II ORION with recursive dependency stability"

# System dependencies with cleanup
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create application directory
WORKDIR /app

# Copy dependency files first for layer caching
COPY ../requirements.txt ./

# Install Python dependencies with validation
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    python -c "import cirq, networkx, fastapi; print('Core dependencies validated')" && \
    python -c "print(f'NetworkX version: {__import__(\"networkx\").__version__}')" && \
    python -c "print(f'Cirq version: {__import__(\"cirq\").__version__}')"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/quantum_cache

# Set environment variables for HFCTM-II
ENV HFCTM_MODE=production
ENV QUANTUM_FALLBACK=true
ENV RECURSIVE_DEPTH=3

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Non-root user for security
RUN useradd -m hfctm && chown -R hfctm:hfctm /app
USER hfctm

# Expose port
EXPOSE 8000

# Start command
CMD ["uvicorn", "orion_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
