-- db/init.sql

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
  id serial PRIMARY KEY,
  content text,
  embedding vector(384),  -- Adjust the dimension to match your model's output
  artwork_name text,
  artist_name text,
  artist_nationality text,
  artist_birth text,
  auction_house_name text,
  auction_location text,
  auction_start_date date,
  price_estimate_min numeric,
  price_estimate_max numeric,
  price_sold numeric,
  currency text,
  artwork_materials text,
  artwork_measurements_width numeric,
  artwork_measurements_height numeric,
  artwork_measurements_unit text
);

-- Create indexes
CREATE INDEX documents_embedding_idx ON documents USING ivfflat (embedding) WITH (lists = 100);
CREATE INDEX documents_artist_name_idx ON documents (artist_name);
CREATE INDEX documents_price_sold_idx ON documents (price_sold);
