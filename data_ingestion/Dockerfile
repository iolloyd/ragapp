# data_ingestion/Dockerfile

FROM python:3.10-slim

WORKDIR /app

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the data ingestion script and data
COPY data_ingestion.py .
# COPY ../data/data.jsonl ./data.jsonl

# The entrypoint will be defined in the docker-compose.yml
