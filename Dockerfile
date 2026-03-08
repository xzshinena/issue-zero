FROM python:3.13-slim

WORKDIR /app

COPY main/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main/ ./main/

EXPOSE 8000

CMD ["uvicorn", "main.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
