# app/Dockerfile

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

# Set environment variables for model names
ARG EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
ARG GENERATION_MODEL_NAME=EleutherAI/gpt-neo-1.3B

# Download the models during build
RUN python -c "from transformers import AutoTokenizer, AutoModel; \
    AutoTokenizer.from_pretrained('${EMBEDDING_MODEL_NAME}'); \
    AutoModel.from_pretrained('${EMBEDDING_MODEL_NAME}'); \
    AutoTokenizer.from_pretrained('${GENERATION_MODEL_NAME}'); \
    from transformers import AutoModelForCausalLM; \
    AutoModelForCausalLM.from_pretrained('${GENERATION_MODEL_NAME}');"

# Copy the Streamlit app
COPY app.py .

# Expose the port that Streamlit uses
EXPOSE 8501

# Run the app
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]
