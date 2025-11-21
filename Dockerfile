# model-service/Dockerfile
FROM python:3.12-slim


WORKDIR /root/sms


COPY requirements.txt .


RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt


COPY . .


ENV MODEL_DIR=/root/sms/output
ENV MODEL_FILENAME=model.joblib
ENV MODEL_URL=
ENV PREPROCESSOR_URL=
ENV MODEL_PORT=8081

EXPOSE ${MODEL_PORT}


CMD ["python", "src/serve_model.py"]