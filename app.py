"""
app.py
Flask API for customer churn prediction. Provides /predict endpoint for JSON input.
"""
from flask import Flask, request, jsonify, render_template
from predict_churn import predict_churn

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        result = predict_churn(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
