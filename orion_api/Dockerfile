# Pin Python version for consistency
FROM python:3.11.7-slim

# Install system dependencies first
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install in stages
COPY ../requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Validate installation
RUN python -c "import cirq, networkx, fastapi; print('Dependencies OK')"

# Copy application code
COPY . .

CMD ["uvicorn", "orion_api.main:app", "--host", "0.0.0.0", "--port", "8080"]
