
import joblib
import os 
import requests 
import time
from flask import Flask, jsonify, request
from flasgger import Swagger
import pandas as pd
from text_preprocessing import prepare, _extract_message_len, _text_process


MODEL_DIR = os.getenv("MODEL_DIR", "/root/sms/output")
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "model.joblib")
MODEL_URL = os.getenv("MODEL_URL", "")
MODEL_PORT = os.getenv("MODEL_PORT", "8081")
PREPROCESSOR_URL = os.getenv("PREPROCESSOR_URL", "")
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
PREPROC_PATH = os.path.join(MODEL_DIR, "preprocessor.joblib")


def download_file(url, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"Downloading {url} -> {dest_path}")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def ensure_asset(path, url, asset_name):

    if os.path.exists(path):
        print(f"{asset_name} already exists: {path}")
        return

    if not url:
        raise RuntimeError(f"{asset_name} missing and no URL provided: {path}")

    print(f"{asset_name} not found. Downloading from {url}")
    download_file(url, path)
    print(f"{asset_name} downloaded.")

def load_assets():
    # 1. Ensure model exists
    ensure_asset(
        path=MODEL_PATH,
        url=MODEL_URL,
        asset_name="MODEL"
    )

    # 2. Ensure preprocessor exists
    ensure_asset(
        path=PREPROC_PATH,
        url=PREPROCESSOR_URL,
        asset_name="PREPROCESSOR"
    )

    # 3. Load both
    print("Loading model...")
    model = joblib.load(MODEL_PATH)
    print("Model loaded.")

    print("Loading preprocessor...")
    preprocessor = joblib.load(PREPROC_PATH)
    print("Preprocessor loaded.")

    return model, preprocessor


try:
    GLOBAL_MODEL, GLOBAL_PREPROC = load_assets()
except Exception as e:
    print(f"FATAL: Could not initialize assets: {e}")
    time.sleep(5)
    raise

# F10 CHANGES END HERE


app = Flask(__name__)
swagger = Swagger(app)

@app.route('/predict', methods=['POST'])
def predict():
    """... (Swagger documentation remains the same) ..."""
    

    if GLOBAL_MODEL is None:
        return jsonify({"error": "Model failed to load during startup."}), 503

    input_data = request.get_json()
    sms = input_data.get('sms')
    processed_sms = GLOBAL_PREPROC.transform([sms])
    
   
    prediction = GLOBAL_MODEL.predict(processed_sms)[0] 
    
    res = {
        "result": prediction,
        "classifier": "decision tree",
        "sms": sms
    }
    print(res)
    return jsonify(res)

if __name__ == '__main__':
    # using the dynamic port from the environment variable
    app.run(host="0.0.0.0", port=int(MODEL_PORT), debug=True)