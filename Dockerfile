FROM python:3.12.9-slim

WORKDIR /root/sms
COPY . .
RUN pip install -r requirements.txt

RUN mkdir -p output && \
    python src/read_data.py && \
    python src/text_preprocessing.py && \
    python src/text_classification.py


ENV SERVER_PORT=8081
EXPOSE ${SERVER_PORT}

CMD ["python", "src/serve_model.py"]
