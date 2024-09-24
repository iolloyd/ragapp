# db/Dockerfile

FROM postgres:15

# Install pgvector extension
RUN apt-get update && apt-get install -y \
    postgresql-server-dev-15 \
    build-essential \
    git

RUN git clone --branch v0.4.2 --depth 1 https://github.com/pgvector/pgvector /tmp/pgvector \
    && cd /tmp/pgvector \
    && make \
    && make install \
    && rm -rf /tmp/pgvector

# Copy initialization scripts
COPY init.sql /docker-entrypoint-initdb.d/
