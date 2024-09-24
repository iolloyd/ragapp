# app/app.py

import streamlit as st
import psycopg2
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import numpy as np
import os

# Database configuration from environment variables
DB_NAME = os.getenv('POSTGRES_DB', 'postgres')
DB_USER = os.getenv('POSTGRES_USER', 'postgres')
DB_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'postgres')
DB_HOST = os.getenv('DB_HOST', 'db')  # Service name defined in docker-compose.yml
DB_PORT = os.getenv('DB_PORT', '5432')

# Model configuration
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', 'sentence-transformers/all-MiniLM-L6-v2')
GENERATION_MODEL_NAME = os.getenv('GENERATION_MODEL_NAME', 'EleutherAI/gpt-neo-1.3B')  # Adjust to your local LLM

# Load the tokenizer and model for embeddings
@st.cache_resource
def load_embedding_model():
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)
    model.eval()
    return tokenizer, model

# Load the tokenizer and model for text generation
@st.cache_resource
def load_generation_model():
    gen_tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL_NAME)
    gen_model = AutoModelForCausalLM.from_pretrained(GENERATION_MODEL_NAME)
    gen_model.eval()
    return gen_tokenizer, gen_model

def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].numpy()[0]
    return embedding

def search_documents(query_embedding, artist_name_filter=None, min_price=None, max_price=None):
    # Database connection
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD,
        host=DB_HOST, port=DB_PORT
    )
    cur = conn.cursor()

    # Build the SQL query
    sql = """
    SELECT content, artwork_name, artist_name, price_sold, currency
    FROM documents
    WHERE 1=1
    """
    params = []

    # Add metadata filters if provided
    if artist_name_filter:
        sql += " AND artist_name ILIKE %s"
        params.append(f"%{artist_name_filter}%")
    if min_price is not None and max_price is not None:
        sql += " AND price_sold BETWEEN %s AND %s"
        params.extend([min_price, max_price])
    elif min_price is not None:
        sql += " AND price_sold >= %s"
        params.append(min_price)
    elif max_price is not None:
        sql += " AND price_sold <= %s"
        params.append(max_price)

    # Add vector similarity condition
    sql += " ORDER BY embedding <=> %s LIMIT 5"
    params.append(query_embedding.tolist())

    # Execute the query
    cur.execute(sql, params)
    results = cur.fetchall()

    cur.close()
    conn.close()
    return results

def generate_answer(question, context_documents):
    gen_tokenizer, gen_model = load_generation_model()

    # Prepare the prompt
    context_text = "\n\n".join([doc[0] for doc in context_documents])  # Combine content fields
    prompt = f"""
You are an expert on artworks and auctions.

Context:
{context_text}

Question:
{question}

Answer:
"""
    # Tokenize and generate the answer
    inputs = gen_tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=1024)
    with torch.no_grad():
        outputs = gen_model.generate(
            inputs,
            max_length=1024,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
    answer = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from the generated text
    answer = answer[len(prompt):].strip()
    return answer

def main():
    st.title("Artworks Search with RAG")

    query_text = st.text_input("Enter your question")
    artist_name_filter = st.text_input("Filter by artist name (optional)")
    min_price = st.number_input("Minimum price (optional)", min_value=0.0, step=0.01)
    max_price = st.number_input("Maximum price (optional)", min_value=0.0, step=0.01)

    if st.button("Get Answer"):
        if not query_text.strip():
            st.warning("Please enter a question.")
            return

        # Generate embedding for the query
        tokenizer, model = load_embedding_model()
        query_embedding = get_embedding(query_text, tokenizer, model)

        # Search for relevant documents
        context_documents = search_documents(query_embedding, artist_name_filter or None,
                                             min_price or None, max_price or None)

        if context_documents:
            # Generate the answer using the local LLM
            answer = generate_answer(query_text, context_documents)
            st.write("**Answer:**")
            st.write(answer)

            st.write("---")
            st.write("**Context Documents:**")
            for content, artwork_name, artist_name, price_sold, currency in context_documents:
                st.subheader(artwork_name)
                st.write(f"**Artist:** {artist_name}")
                st.write(f"**Price Sold:** {price_sold} {currency}")
                st.write(f"**Description:** {content}")
                st.write("---")
        else:
            st.info("No relevant documents found.")

if __name__ == '__main__':
    main()
