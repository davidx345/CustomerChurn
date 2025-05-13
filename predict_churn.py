"""
predict_churn.py
Helper functions for input validation and churn prediction using the saved model.
"""
import joblib
import numpy as np

# Load model and preprocessors
model_bundle = joblib.load('churn_model.pkl')
model = model_bundle['model']
scaler = model_bundle['scaler']
le_gender = model_bundle['le_gender']
le_geo = model_bundle['le_geo']
features = model_bundle['features']

def validate_input(data):
    """
    Validate and preprocess input data for prediction.
    Returns a numpy array ready for model prediction or raises ValueError.
    """
    required = set(features)
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"Missing fields: {', '.join(missing)}")
    # Copy and encode
    x = []
    for f in features:
        val = data[f]
        if f == 'Gender':
            val = le_gender.transform([val])[0]
        elif f == 'Geography':
            val = le_geo.transform([val])[0]
        x.append(val)
    # Scale
    x_scaled = scaler.transform([x])
    return x_scaled

def predict_churn(data):
    """
    Predict churn (Yes/No) and probability for a single customer dict.
    Returns: dict with 'prediction' and 'probability'.
    """
    x = validate_input(data)
    proba = model.predict_proba(x)[0, 1]
    pred = model.predict(x)[0]
    return {
        'prediction': 'Yes' if pred == 1 else 'No',
        'probability': float(proba)
    }
