#!/bin/sh
set -e

minio server /data --console-address ":${MINIO_CONSOLE_PORT}" &
MINIO_PID=$!

# until mc ready local; do
#   sleep 1
# done
until mc alias set minio http://127.0.0.1:$MINIO_PORT "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD" >/dev/null 2>&1; do
  sleep 1
done

mc mb -p minio/$MLFLOW_ARTFACT_BUCKET

wait $MINIO_PID
