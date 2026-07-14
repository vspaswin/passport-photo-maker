FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY cli.py .
COPY LICENSE README.md ./

ENV APP_ENV=production \
    HOST=0.0.0.0 \
    PORT=8765 \
    PYTHONUNBUFFERED=1

EXPOSE 8765

# Persist jobs/credits outside the container with a volume on /root/.passport-photo-maker
CMD ["python", "-m", "app.main"]
