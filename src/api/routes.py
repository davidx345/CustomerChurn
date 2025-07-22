# src/api/routes.py
"""
API routes for customer churn prediction service
[Intent] RESTful API with proper error handling and validation
[DevOps] Structured responses for monitoring and debugging
[Security] Input validation and rate limiting ready
"""

from flask import Blueprint, request, jsonify, send_file
from werkzeug.utils import secure_filename
import pandas as pd
import io
import os
from typing import Dict, Any
from src.models.churn_predictor import ChurnPredictor
from src.utils.logging_config import api_logger
from src.utils.metrics import track_request_metrics, track_batch_size, get_metrics
from config.settings import get_config

config = get_config()
api_bp = Blueprint('api', __name__)

# Initialize predictor
predictor = ChurnPredictor()

@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for load balancers and monitoring"""
    try:
        model_info = predictor.get_model_info()
        return jsonify({
            'status': 'healthy',
            'service': 'customer-churn-prediction',
            'version': '2.0.0',
            'model_loaded': model_info['model_type'] != 'Unknown',
            'timestamp': pd.Timestamp.now().isoformat()
        }), 200
    except Exception as e:
        api_logger.error("Health check failed", error=str(e))
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': pd.Timestamp.now().isoformat()
        }), 503

@api_bp.route('/predict', methods=['POST'])
@track_request_metrics('predict')
def predict_single():
    """Predict churn for a single customer"""
    try:
        data = request.get_json()
        
        if not data:
            api_logger.warning("Empty request body received")
            return jsonify({
                'error': 'No input data provided',
                'code': 'EMPTY_REQUEST'
            }), 400
        
        if not isinstance(data, dict):
            api_logger.warning("Invalid request format", data_type=str(type(data)))
            return jsonify({
                'error': 'Invalid data format. Expected JSON object.',
                'code': 'INVALID_FORMAT'
            }), 400
        
        api_logger.info("Single prediction request received", 
                       customer_age=data.get('Age'), 
                       geography=data.get('Geography'))
        
        result = predictor.predict(data)
        
        if 'error' in result:
            return jsonify(result), 400
        
        # Add request metadata
        result['request_id'] = request.headers.get('X-Request-ID', 'unknown')
        result['api_version'] = '2.0.0'
        
        api_logger.info("Single prediction completed", 
                       prediction=result.get('prediction'),
                       probability=result.get('probability'))
        
        return jsonify(result), 200
        
    except Exception as e:
        api_logger.error("Single prediction failed", error=str(e))
        return jsonify({
            'error': 'Internal server error during prediction',
            'code': 'PREDICTION_ERROR',
            'details': str(e) if config.DEBUG else None
        }), 500

@api_bp.route('/batch_predict', methods=['POST'])
@track_request_metrics('batch_predict')
def predict_batch():
    """Predict churn for multiple customers via CSV upload"""
    try:
        if 'csvFile' not in request.files:
            api_logger.warning("No CSV file in batch request")
            return jsonify({
                'error': 'No CSV file provided',
                'code': 'NO_FILE'
            }), 400
        
        file = request.files['csvFile']
        
        if file.filename == '':
            api_logger.warning("Empty filename in batch request")
            return jsonify({
                'error': 'No file selected',
                'code': 'EMPTY_FILENAME'
            }), 400
        
        if not file.filename.lower().endswith('.csv'):
            api_logger.warning("Invalid file type", filename=file.filename)
            return jsonify({
                'error': 'Invalid file type. Please upload a CSV file.',
                'code': 'INVALID_FILE_TYPE'
            }), 400
        
        # Track file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        track_batch_size(file_size)
        
        # Check file size limit
        if file_size > config.MAX_CONTENT_LENGTH:
            api_logger.warning("File too large", file_size=file_size)
            return jsonify({
                'error': f'File too large. Maximum size: {config.MAX_CONTENT_LENGTH / (1024*1024):.1f}MB',
                'code': 'FILE_TOO_LARGE'
            }), 413
        
        try:
            df = pd.read_csv(file)
        except Exception as e:
            api_logger.error("Failed to read CSV", error=str(e))
            return jsonify({
                'error': f'Error reading CSV file: {str(e)}',
                'code': 'CSV_READ_ERROR'
            }), 400
        
        if df.empty:
            api_logger.warning("Empty CSV file uploaded")
            return jsonify({
                'error': 'CSV file is empty',
                'code': 'EMPTY_FILE'
            }), 400
        
        api_logger.info("Batch prediction started", row_count=len(df))
        
        predictions = []
        error_count = 0
        
        for idx, row in df.iterrows():
            try:
                data_dict = row.to_dict()
                
                # Clean and convert data types
                for key in ['CreditScore', 'Age', 'Tenure', 'NumOfProducts', 'HasCrCard', 'IsActiveMember']:
                    if key in data_dict and pd.notna(data_dict[key]):
                        data_dict[key] = int(float(data_dict[key]))
                
                for key in ['Balance', 'EstimatedSalary']:
                    if key in data_dict and pd.notna(data_dict[key]):
                        data_dict[key] = float(data_dict[key])
                
                prediction_result = predictor.predict(data_dict)
                
                # Create output row
                output_row = row.to_dict()
                output_row['ChurnPrediction'] = prediction_result.get('prediction', 'Error')
                output_row['ChurnProbability'] = prediction_result.get('probability', 'Error')
                output_row['HarmfulnessFlag'] = prediction_result.get('harmfulness_flag', False)
                output_row['BiasFlags'] = str(prediction_result.get('bias_flags', {}))
                
                predictions.append(output_row)
                
            except Exception as e:
                error_count += 1
                error_row = row.to_dict()
                error_row['ChurnPrediction'] = 'Error'
                error_row['ChurnProbability'] = f'Error: {str(e)}'
                error_row['HarmfulnessFlag'] = False
                error_row['BiasFlags'] = ''
                predictions.append(error_row)
                
                api_logger.warning("Row prediction failed", row_index=idx, error=str(e))
        
        output_df = pd.DataFrame(predictions)
        
        # Create CSV in memory
        csv_buffer = io.StringIO()
        output_df.to_csv(csv_buffer, index=False)
        
        mem_file = io.BytesIO()
        mem_file.write(csv_buffer.getvalue().encode('utf-8'))
        mem_file.seek(0)
        csv_buffer.close()
        
        api_logger.info("Batch prediction completed", 
                       total_rows=len(df), 
                       error_count=error_count,
                       success_rate=f"{((len(df) - error_count) / len(df) * 100):.1f}%")
        
        return send_file(
            mem_file,
            mimetype='text/csv',
            as_attachment=True,
            download_name='batch_predictions.csv'
        )
        
    except Exception as e:
        api_logger.error("Batch prediction failed", error=str(e))
        return jsonify({
            'error': 'Internal server error during batch prediction',
            'code': 'BATCH_ERROR',
            'details': str(e) if config.DEBUG else None
        }), 500

@api_bp.route('/api/feature_importance', methods=['GET'])
@track_request_metrics('feature_importance')
def get_feature_importance():
    """Get global feature importance from the model"""
    try:
        model_info = predictor.get_model_info()
        
        if predictor.model and hasattr(predictor.model, 'feature_importances_'):
            importances = predictor.model.feature_importances_
            features = model_info.get('features', [])
            
            if len(features) == len(importances):
                sorted_importance = sorted(
                    zip(features, importances), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                return jsonify({
                    'features': [item[0] for item in sorted_importance],
                    'importances': [float(item[1]) for item in sorted_importance],
                    'model_type': model_info['model_type']
                }), 200
            else:
                api_logger.warning("Feature name/importance count mismatch")
                return jsonify({
                    'error': 'Feature names and importances length mismatch',
                    'code': 'FEATURE_MISMATCH'
                }), 500
        else:
            api_logger.warning("Model has no feature_importances_ attribute")
            return jsonify({
                'error': 'Model does not support feature importance',
                'code': 'NO_FEATURE_IMPORTANCE'
            }), 501
            
    except Exception as e:
        api_logger.error("Feature importance request failed", error=str(e))
        return jsonify({
            'error': 'Failed to get feature importance',
            'code': 'FEATURE_IMPORTANCE_ERROR'
        }), 500

@api_bp.route('/api/model_info', methods=['GET'])
def get_model_info():
    """Get model information and metadata"""
    try:
        model_info = predictor.get_model_info()
        return jsonify(model_info), 200
    except Exception as e:
        api_logger.error("Model info request failed", error=str(e))
        return jsonify({
            'error': 'Failed to get model information',
            'code': 'MODEL_INFO_ERROR'
        }), 500

@api_bp.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint"""
    try:
        return get_metrics(), 200, {'Content-Type': 'text/plain; charset=utf-8'}
    except Exception as e:
        api_logger.error("Metrics request failed", error=str(e))
        return "# Metrics unavailable\n", 500, {'Content-Type': 'text/plain; charset=utf-8'}
