FROM python:3.12.9-slim

WORKDIR /root/sms
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8081
CMD ["python", "./artifact_bundle/serve_model.py"]
