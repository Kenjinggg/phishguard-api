import joblib
import numpy as np
import json
import os
from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
from config import API_KEY, RATE_LIMIT

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

# ─── Rate Limiting ────────────────────────────────────────────────────────────
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[RATE_LIMIT]
)

# ─── Load Model, Scaler, and Feature List ────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'rf_model.joblib')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'model', 'scaler.joblib')
FEATURES_PATH = os.path.join(os.path.dirname(__file__), 'model', 'continuous_features.json')

try:
    model = joblib.load(MODEL_PATH)
    print(f"[OK] Model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    model = None

try:
    scaler = joblib.load(SCALER_PATH)
    print(f"[OK] Scaler loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load scaler: {e}")
    scaler = None

try:
    with open(FEATURES_PATH, 'r') as f:
        continuous_features = json.load(f)
    print(f"[OK] Continuous features loaded. Count: {len(continuous_features)}")
except Exception as e:
    print(f"[ERROR] Failed to load continuous features: {e}")
    continuous_features = []

# ─── Expected Feature Order ───────────────────────────────────────────────────
FEATURE_COLUMNS = [
    'URLLength', 'DomainLength', 'IsDomainIP', 'URLSimilarityIndex',
    'CharContinuationRate', 'TLDLegitimateProb', 'URLCharProb', 'TLDLength',
    'NoOfSubDomain', 'HasObfuscation', 'NoOfObfuscatedChar', 'ObfuscationRatio',
    'NoOfLettersInURL', 'LetterRatioInURL', 'NoOfDegitsInURL', 'DegitRatioInURL',
    'NoOfEqualsInURL', 'NoOfQMarkInURL', 'NoOfAmpersandInURL',
    'NoOfOtherSpecialCharsInURL', 'SpacialCharRatioInURL', 'IsHTTPS',
    'HasTitle', 'DomainTitleMatchScore', 'URLTitleMatchScore', 'HasFavicon',
    'Robots', 'IsResponsive', 'HasDescription', 'NoOfPopup', 'NoOfiFrame',
    'HasExternalFormSubmit', 'HasSocialNet', 'HasSubmitButton', 'HasHiddenFields',
    'HasPasswordField', 'Bank', 'Pay', 'Crypto', 'HasCopyrightInfo',
    'NoOfImage', 'NoOfCSS', 'NoOfJS', 'NoOfSelfRef', 'NoOfEmptyRef',
    'NoOfExternalRef'
]

# ─── Helper: Validate API Key ─────────────────────────────────────────────────
def is_valid_api_key(req):
    key = req.headers.get('X-API-Key')
    return key == API_KEY

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "running",
        "message": "PhishGuard ML API is online",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }), 200


@app.route('/predict', methods=['POST'])
@limiter.limit(RATE_LIMIT)
def predict():

    # 1. Validate API key
    if not is_valid_api_key(request):
        return jsonify({
            "error": "Unauthorized. Invalid or missing API key."
        }), 401

    # 2. Validate request has JSON body
    if not request.is_json:
        return jsonify({
            "error": "Request body must be JSON."
        }), 400

    data = request.get_json()

    # 3. Validate all required features are present
    missing_features = [f for f in FEATURE_COLUMNS if f not in data]
    if missing_features:
        return jsonify({
            "error": "Missing required features.",
            "missing": missing_features
        }), 400

    # 4. Check model and scaler are loaded
    if model is None:
        return jsonify({
            "error": "Model is not loaded. Contact administrator."
        }), 500

    if scaler is None:
        return jsonify({
            "error": "Scaler is not loaded. Contact administrator."
        }), 500

    # 5. Build feature array in correct column order
    try:
        feature_values = {col: float(data[col]) for col in FEATURE_COLUMNS}
    except (ValueError, TypeError) as e:
        return jsonify({
            "error": f"Invalid feature value. All features must be numeric. Detail: {str(e)}"
        }), 400

    # 6. Apply scaling to continuous features only
    try:
        import pandas as pd
        features_df = pd.DataFrame([feature_values])

        # Scale only the continuous features
        features_df[continuous_features] = scaler.transform(
            features_df[continuous_features]
        )

        features_array = features_df[FEATURE_COLUMNS].values

    except Exception as e:
        return jsonify({
            "error": f"Feature scaling failed. Detail: {str(e)}"
        }), 500

    # 7. Run prediction
    try:
        prediction = model.predict(features_array)[0]
        probabilities = model.predict_proba(features_array)[0]

        phishing_probability = float(probabilities[0])
        legitimate_probability = float(probabilities[1])

        return jsonify({
            "prediction": int(prediction),
            "label": "legitimate" if prediction == 1 else "phishing",
            "phishing_probability": round(phishing_probability, 4),
            "legitimate_probability": round(legitimate_probability, 4)
        }), 200

    except Exception as e:
        return jsonify({
            "error": f"Prediction failed. Detail: {str(e)}"
        }), 500


# ─── Error Handlers ───────────────────────────────────────────────────────────

@app.errorhandler(429)
def rate_limit_exceeded(e):
    return jsonify({
        "error": "Rate limit exceeded. Too many requests. Please slow down."
    }), 429

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "error": "Endpoint not found."
    }), 404


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)