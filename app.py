import joblib
import numpy as np
import json
import os
import logging
import hmac
import math
from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
from config import API_KEY, RATE_LIMIT

# ─── Logging Setup ────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error("Failed to load model: %s", e)
    model = None

try:
    scaler = joblib.load(SCALER_PATH)
    logger.info("Scaler loaded successfully.")
except Exception as e:
    logger.error("Failed to load scaler: %s", e)
    scaler = None

try:
    with open(FEATURES_PATH, 'r') as f:
        continuous_features = json.load(f)
    logger.info("Continuous features loaded. Count: %d", len(continuous_features))
except Exception as e:
    logger.error("Failed to load continuous features: %s", e)
    continuous_features = []

# ─── Verify Model Classes ────────────────────────────────────────────────────
if model is not None:
    expected_classes = [0, 1]
    actual_classes = list(model.classes_)
    if actual_classes != expected_classes:
        logger.warning(
            "Model classes mismatch! Expected %s but got %s. "
            "Class indexing may be incorrect.",
            expected_classes, actual_classes
        )
    else:
        logger.info("Model classes verified: %s (0=phishing, 1=legitimate)", actual_classes)

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
    key = req.headers.get('X-API-Key') or ''
    return hmac.compare_digest(key, API_KEY)

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
        logger.error("Invalid feature value received: %s", e)
        return jsonify({
            "error": "Invalid feature value. All features must be numeric."
        }), 400

    # 5b. Validate no NaN or Inf values
    for col, val in feature_values.items():
        if math.isnan(val) or math.isinf(val):
            return jsonify({
                "error": f"Invalid feature value for '{col}': NaN or Inf values are not allowed."
            }), 400

    # 6. Apply scaling to continuous features only
    try:
        feature_array = np.array([[feature_values[col] for col in FEATURE_COLUMNS]])
        cont_indices = [FEATURE_COLUMNS.index(f) for f in continuous_features]
        feature_array[0, cont_indices] = scaler.transform(feature_array[:, cont_indices])[0]

    except Exception as e:
        logger.exception("Feature scaling failed.")
        return jsonify({
            "error": "Feature scaling failed."
        }), 500

    # 7. Run prediction
    try:
        prediction = model.predict(feature_array)[0]
        probabilities = model.predict_proba(feature_array)[0]

        classes = list(model.classes_)
        phishing_idx = classes.index(0)
        legitimate_idx = classes.index(1)
        phishing_probability = float(probabilities[phishing_idx])
        legitimate_probability = float(probabilities[legitimate_idx])

        return jsonify({
            "prediction": int(prediction),
            "label": "legitimate" if prediction == 1 else "phishing",
            "phishing_probability": round(phishing_probability, 4),
            "legitimate_probability": round(legitimate_probability, 4)
        }), 200

    except Exception as e:
        logger.exception("Prediction failed.")
        return jsonify({
            "error": "Prediction failed."
        }), 500


# ─── Error Handlers ───────────────────────────────────────────────────────────

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed."}), 405

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

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error."}), 500


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', '5000'))
    app.run(debug=debug, host=host, port=port, use_reloader=False)