#!/bin/bash
set -e

# Start ParadeDB (original entrypoint)
exec /usr/local/bin/docker-entrypoint.sh postgres &

# Wait for DB to accept connections
until pg_isready -h localhost -p 5432; do
  sleep 1
done

echo "Postgres ready"

# Start backup API
uvicorn backup_api:app --host 0.0.0.0 --port 8001
