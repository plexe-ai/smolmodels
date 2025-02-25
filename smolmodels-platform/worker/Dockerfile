FROM python:3.12

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Create .smolcache directory and set permissions
RUN mkdir -p /app/.smolcache && chmod 777 /app/.smolcache

# Install Poetry using pip (simpler than curl method)
RUN pip install poetry==1.8.2

# Copy only pyproject.toml and poetry.lock first to leverage Docker cache
COPY pyproject.toml poetry.lock* ./

# Configure poetry to not use a virtual environment inside the Docker container
RUN poetry config virtualenvs.create false

# Install project dependencies
RUN poetry install --no-dev --no-interaction

# Copy application code
COPY . .

# Set environment variable for Python module paths
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/.smolcache

# Start worker process
CMD ["python", "-m", "worker.main"]
