FROM python:3.11-slim

WORKDIR /app
COPY . .
# RUN apt-get update && \
#     apt-get install -y gcc portaudio19-dev libasound-dev libsndfile1-dev && \
#     rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
