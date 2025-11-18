# src/serve_model.py
import joblib
import os # Import the os module
import requests # Need to install the 'requests' library
import time
from flask import Flask, jsonify, request
from flasgger import Swagger
import pandas as pd
from text_preprocessing import prepare, _extract_message_len, _text_process

# --- F10 CHANGES START HERE ---
MODEL_DIR = os.getenv("MODEL_DIR", "/app/output")
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "model.joblib")
MODEL_URL = os.getenv("MODEL_URL", "")
MODEL_PORT = os.getenv("MODEL_PORT", "8081")

MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

def load_or_download_model():
    """Loads model from volume mount or downloads it."""
    # 1. Check for volume-mounted model (F10 requirement)
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from volume mount: {MODEL_PATH}")
        return joblib.load(MODEL_PATH)

    # 2. Check if a download URL is provided and download it (F10 requirement)
    if MODEL_URL:
        print(f"Model not found. Downloading from: {MODEL_URL}")
        try:
            # Ensure the directory exists
            os.makedirs(MODEL_DIR, exist_ok=True)
            
            # Download the file
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print("Download successful. Loading model.")
            return joblib.load(MODEL_PATH)
        except Exception as e:
            print(f"FATAL: Failed to download or load model: {e}")
            # If model is required to run, exit the process
            time.sleep(10) # Wait to allow logs to be viewed before crashing
            exit(1)
    
    # 3. If no model is mounted and no URL is provided (error state)
    print("FATAL: No model found and no MODEL_URL provided. Exiting.")
    time.sleep(10)
    exit(1)


# Load the model once when the application starts
# This makes it global and avoids reloading it on every request.
try:
    GLOBAL_MODEL = load_or_download_model()
except Exception as e:
    # A failure here means the model could not be loaded on startup
    print(f"Application failed to initialize model: {e}")
    GLOBAL_MODEL = None # Set to None, error will be caught later.

# --- F10 CHANGES END HERE ---


app = Flask(__name__)
swagger = Swagger(app)

@app.route('/predict', methods=['POST'])
def predict():
    """... (Swagger documentation remains the same) ..."""
    
    # Check if the model failed to load at startup
    if GLOBAL_MODEL is None:
        return jsonify({"error": "Model failed to load during startup."}), 503

    input_data = request.get_json()
    sms = input_data.get('sms')
    processed_sms = prepare(sms)
    
    # Use the globally loaded model instead of joblib.load('output/model.joblib')
    prediction = GLOBAL_MODEL.predict(processed_sms)[0] 
    
    res = {
        "result": prediction,
        "classifier": "decision tree",
        "sms": sms
    }
    print(res)
    return jsonify(res)

if __name__ == '__main__':
    # Use the dynamic port from the environment variable
    app.run(host="0.0.0.0", port=int(MODEL_PORT), debug=True)