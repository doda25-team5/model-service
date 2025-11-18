# model-service/Dockerfile
FROM python:3.12-slim

# working dir inside container
WORKDIR /app

# copy only requirements first for caching
COPY requirements.txt .

# upgrade pip and install requirements
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

# copy the source code
COPY . .

# environment defaults (override at runtime)
ENV MODEL_DIR=/app/output
ENV MODEL_FILENAME=model.joblib
ENV MODEL_URL=""         # URL to download model if none is mounted; leave blank to require mount
ENV MODEL_PORT=8081

EXPOSE 8081

# start the service (the Python script handles model loading/downloading)
CMD ["python", "src/serve_model.py"]
