# data_ingestion/data_ingestion.py

import json
import psycopg2
import torch
from transformers import AutoTokenizer, AutoModel
from datetime import datetime
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)

# Database configuration from environment variables
DB_NAME = os.getenv('POSTGRES_DB', 'postgres')
DB_USER = os.getenv('POSTGRES_USER', 'postgres')
DB_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'postgres')
DB_HOST = os.getenv('DB_HOST', 'db')  # Service name defined in docker-compose.yml
DB_PORT = os.getenv('DB_PORT', '5432')

# Model configuration
MODEL_NAME = os.getenv('MODEL_NAME', 'sentence-transformers/all-MiniLM-L6-v2')

def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].numpy()[0]
    return embedding

def parse_date(date_str):
    if not date_str:
        return None
    for fmt in ('%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y'):
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return None  # Return None if date format is unrecognized

def main():
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    # Database connection
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD,
            host=DB_HOST, port=DB_PORT
        )
        cur = conn.cursor()
    except Exception as e:
        logging.error(f"Database connection failed: {e}")
        return

    # Read and process the JSONL file
    try:
        with open('data/data.jsonl', 'r') as file:
            for line in file:
                record = json.loads(line)

                # Combine text fields for embedding
                content_fields = [
                    record.get('artwork_name', ''),
                    record.get('artwork_description', ''),
                    record.get('artwork_materials', ''),
                    record.get('artwork_exhibited', ''),
                    record.get('artwork_literature', '')
                ]
                content = ' '.join(filter(None, content_fields))
                if not content.strip():
                    continue  # Skip records with no content

                # Extract fields
                artwork_name = record.get('artwork_name', '')
                artist_name = record.get('artist_name', '')
                artist_nationality = record.get('artist_nationality', '')
                artist_birth = record.get('artist_birth', '')
                auction_house_name = record.get('auction_house_name', '')
                auction_location = record.get('auction_location', '')
                auction_start_date = parse_date(record.get('auction_start_date'))
                price_estimate_min = record.get('price_estimate_min')
                price_estimate_max = record.get('price_estimate_max')
                price_sold = record.get('price_sold')
                currency = record.get('currency', '')
                artwork_materials = record.get('artwork_materials', '')
                artwork_measurements_width = record.get('artwork_measurements_width')
                artwork_measurements_height = record.get('artwork_measurements_height')
                artwork_measurements_unit = record.get('artwork_measurements_unit', '')

                # Generate embedding
                embedding = get_embedding(content, tokenizer, model)

                # Insert into PostgreSQL
                cur.execute("""
                    INSERT INTO documents (
                        content, embedding, artwork_name, artist_name, artist_nationality,
                        artist_birth, auction_house_name, auction_location, auction_start_date,
                        price_estimate_min, price_estimate_max, price_sold, currency,
                        artwork_materials, artwork_measurements_width, artwork_measurements_height,
                        artwork_measurements_unit
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    content, embedding.tolist(), artwork_name, artist_name, artist_nationality,
                    artist_birth, auction_house_name, auction_location, auction_start_date,
                    price_estimate_min, price_estimate_max, price_sold, currency,
                    artwork_materials, artwork_measurements_width, artwork_measurements_height,
                    artwork_measurements_unit
                ))
                conn.commit()
                logging.info(f"Inserted record for artwork: {artwork_name}")

    except Exception as e:
        logging.error(f"An error occurred during data processing: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

if __name__ == '__main__':
    main()
