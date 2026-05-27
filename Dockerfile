# Stage 1: build frontend (skip for API-only: docker build --target api .)
# Requires main/frontend/ to be scaffolded first (deferred — see TODOS.md).
FROM node:20-slim AS frontend-builder
WORKDIR /frontend
COPY main/frontend/package*.json ./
RUN npm ci --omit=dev
COPY main/frontend/ ./
RUN npm run build

# Stage 2: Python API
FROM python:3.13-slim AS api
WORKDIR /app

COPY main/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main/ ./main/
COPY --from=frontend-builder /frontend/dist ./main/frontend/dist

EXPOSE 8000
CMD ["uvicorn", "main.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
