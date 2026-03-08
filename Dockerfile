FROM python:3.13-slim

WORKDIR /app

COPY main/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main/ ./main/
COPY frontend/ ./frontend/

ENV PYTHONPATH=/app/main

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
