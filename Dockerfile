FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p logs data/raw data/processed saved_models/ml saved_models/dl

EXPOSE 8000

ENV PYTHONPATH=/app

CMD ["python", "src/api/main.py"]