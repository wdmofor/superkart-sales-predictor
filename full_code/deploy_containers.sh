#!/usr/bin/env bash
# Builds both images, starts them on the superkart-net network, waits for the
# health checks, and sends one prediction through the backend.
# Run it from any folder. Existing containers with the same names are replaced.
set -euo pipefail

cd "$(dirname "$0")"

NETWORK=superkart-net
BACKEND=superkart-full-backend
FRONTEND=superkart-full-frontend

# In a Codespace the frontend calls the backend's public forwarded link, so every request goes to the deployed API.
# Locally it uses the backend's Docker-network name.
if [[ -n "${CODESPACE_NAME:-}" ]]; then
  API_URL="https://${CODESPACE_NAME}-7860.${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN:-app.github.dev}"
else
  API_URL="http://$BACKEND:7860"
fi
echo "Frontend will call the backend at: $API_URL"

docker info > /dev/null
docker network inspect "$NETWORK" > /dev/null 2>&1 || docker network create "$NETWORK"
docker rm -f "$BACKEND" "$FRONTEND" > /dev/null 2>&1 || true

# --no-cache rebuilds every layer, so the running images always match the files in this folder.
docker build --no-cache -t superkart-full-backend:local backend_files
docker build --no-cache -t superkart-full-frontend:local frontend_files

docker run -d --name "$BACKEND" --network "$NETWORK" \
  -p 127.0.0.1:7860:7860 superkart-full-backend:local > /dev/null
docker run -d --name "$FRONTEND" --network "$NETWORK" \
  -p 127.0.0.1:7870:7860 -e "SUPERKART_API_URL=$API_URL" \
  superkart-full-frontend:local > /dev/null

wait_for_health() {
  local name=$1
  local status=unknown
  for _ in $(seq 1 80); do
    status=$(docker inspect -f '{{.State.Health.Status}}' "$name" 2>/dev/null || echo unknown)
    if [[ $status == healthy ]]; then
      echo "$name: healthy"
      return 0
    fi
    sleep 3
  done
  echo "$name did not become healthy (last status: $status)" >&2
  return 1
}

wait_for_health "$BACKEND"
wait_for_health "$FRONTEND"

echo "Smoke test (model prediction for the course template payload; 2904.05 with the shipped model):"
curl -fsS -X POST http://127.0.0.1:7860/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"Product_Weight": 12.66, "Product_Sugar_Content": "Low Sugar", "Product_Allocated_Area": 0.027, "Product_MRP": 117.08, "Store_Size": "Medium", "Store_Location_City_Type": "Tier 2", "Store_Type": "Supermarket Type2", "Product_Id_char": "FD", "Store_Age_Years": 16, "Product_Type_Category": "Non Perishables"}'
echo

docker ps --filter "name=superkart-full" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
