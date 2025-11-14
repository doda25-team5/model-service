FROM python:3.12.9-slim

WORKDIR /root/sms
COPY . .
RUN pip install -r requirements.txt

RUN mkdir -p output && \
    python src/read_data.py && \
    python src/text_preprocessing.py && \
    python src/text_classification.py

EXPOSE 8081
CMD ["python", "src/serve_model.py"]
