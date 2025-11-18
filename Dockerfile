FROM python:3.12.9-slim

WORKDIR /root/sms
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

COPY models/model.pkl ./models/model.pkl

EXPOSE 8081
CMD ["python", "src/serve_model.py"]
