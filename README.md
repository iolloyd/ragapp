# RAG Application with PostgreSQL, pgvector, and Streamlit

This project sets up a Retrieval-Augmented Generation (RAG) application using:

- **PostgreSQL** with **pgvector** extension for vector similarity search.
- **Data Ingestion Service** to process data and store embeddings.
- **Streamlit App** as the frontend interface.
- **Local LLMs** for embedding generation and text generation.

## **Prerequisites**

- **Docker** and **Docker Compose** installed on your machine.

## **Getting Started**

### **1. Place Your Data**

- Place your `data.jsonl` file in the `data/` directory.

### **2. Build and Run the Services**

From the root of your project directory, run:

```bash
docker-compose build
docker-compose up
