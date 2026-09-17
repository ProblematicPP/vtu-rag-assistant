#!/bin/sh
# Creates the extra databases listed in EXTRA_DATABASES (comma-separated),
# owned by POSTGRES_USER. Runs once, on first initialisation of the volume.
set -eu

if [ -z "${EXTRA_DATABASES:-}" ]; then
  exit 0
fi

for db in $(echo "$EXTRA_DATABASES" | tr ',' ' '); do
  echo "Creating database: $db"
  psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    SELECT 'CREATE DATABASE "$db" OWNER "$POSTGRES_USER"'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$db')\gexec
EOSQL
done
