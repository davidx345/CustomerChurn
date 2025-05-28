"""
app.py
Flask API for customer churn prediction. Provides /predict endpoint for JSON input.
"""
from flask import Flask, request, jsonify, render_template, send_file
from predict_churn import predict_churn, get_feature_names_after_preprocessing # Import the new function
import pandas as pd
import io
import joblib # For loading the model bundle to get feature importances

app = Flask(__name__)

# Load model bundle once to get model, explainer, and feature names for global importance
try:
    model_bundle = joblib.load('churn_model.pkl')
    model = model_bundle['model'] # Assumes 'model' key exists
    explainer = model_bundle.get('explainer') # Attempt to load explainer, None if not present
    # Assuming 'features' in the bundle are the original feature names
    # and get_feature_names_after_preprocessing() gives the names model uses
    # For global feature importance, we need names corresponding to model.feature_importances_
    # This usually means the names *after* any encoding/transformations if the model object itself
    # doesn't store them in a directly interpretable way with feature_importances_
    # Let's use get_feature_names_after_preprocessing() as the source of truth for model feature names
    model_feature_names = get_feature_names_after_preprocessing()
    performance_metrics = model_bundle.get('performance_metrics') # Load performance metrics
except Exception as e:
    print(f"Error loading model bundle: {e}") # Generalize error message
    model = None
    explainer = None # Ensure explainer is None if an error occurs during loading
    model_feature_names = []
    performance_metrics = None # Ensure performance_metrics is None if an error occurs

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        if not isinstance(data, dict):
            return jsonify({'error': 'Invalid data format for single prediction. Expected a JSON object.'}), 400
        
        # predict_churn now returns shap_values as well
        result = predict_churn(data) 
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/feature_importance', methods=['GET'])
def feature_importance():
    if model and hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        # Ensure model_feature_names has the same length as importances
        if len(model_feature_names) == len(importances):
            sorted_importance = sorted(zip(model_feature_names, importances), key=lambda x: x[1], reverse=True)
            return jsonify({'features': [item[0] for item in sorted_importance], 
                            'importances': [item[1] for item in sorted_importance]})
        else:
            # Fallback if names don't align: return raw importances if names are problematic
            # This indicates a mismatch that needs fixing in get_feature_names_after_preprocessing or model loading
            print(f"Warning: Mismatch between number of feature names ({len(model_feature_names)}) and importances ({len(importances)}).")
            # Attempt to use 'features' from bundle if model_feature_names is the issue
            bundle_features = model_bundle.get('features')
            if bundle_features and len(bundle_features) == len(importances):
                 sorted_importance = sorted(zip(bundle_features, importances), key=lambda x: x[1], reverse=True)
                 return jsonify({'features': [item[0] for item in sorted_importance], 
                                 'importances': [item[1] for item in sorted_importance]})
            return jsonify({'error': 'Feature names and importances length mismatch.', 'num_names': len(model_feature_names), 'num_importances': len(importances)}), 500

    elif model and explainer: # If we have SHAP explainer, can use mean abs SHAP values
        # This requires a sample of data to calculate global importance.
        # For simplicity, this part is omitted for now as it requires X_train_processed_sample
        # and can be slow for KernelExplainer.
        # If you have X_train_processed_sample, you could do:
        # shap_values_sample = explainer.shap_values(X_train_processed_sample)
        # global_importances = np.abs(shap_values_sample[1]).mean(0) # for churn class
        # sorted_importance = sorted(zip(model_feature_names, global_importances), key=lambda x: x[1], reverse=True)
        # return jsonify({'features': [item[0] for item in sorted_importance], 
        #                 'importances': [item[1] for item in sorted_importance]})
        return jsonify({'error': 'Feature importances from SHAP not implemented yet without a data sample.'}), 501
    else:
        return jsonify({'error': 'Model does not have feature_importances_ attribute or SHAP explainer not available.'}), 501

@app.route('/api/model_performance', methods=['GET'])
def model_performance():
    global performance_metrics # Ensure this is at the top
    if performance_metrics:
        return jsonify(performance_metrics)
    else:
        # Try to load them again if they weren't loaded at startup, though ideally they should be.
        try:
            bundle = joblib.load('churn_model.pkl')
            metrics = bundle.get('performance_metrics')
            if metrics:
                # Cache them for next time if loaded successfully
                performance_metrics = metrics # Assign to the global variable (already declared global)
                return jsonify(metrics)
            else:
                return jsonify({'error': 'Performance metrics not found in model bundle.'}), 404
        except Exception as e:
            return jsonify({'error': f'Could not load performance metrics: {str(e)}'}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        if 'csvFile' not in request.files:
            return jsonify({'error': 'No CSV file provided'}), 400
        
        file = request.files['csvFile']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file and file.filename.endswith('.csv'):
            try:
                df = pd.read_csv(file)
            except Exception as e:
                return jsonify({'error': f'Error reading CSV file: {str(e)}'}), 400

            if df.empty:
                return jsonify({'error': 'CSV file is empty'}), 400

            predictions = []
            # Check if required columns are present (optional, but good practice)
            # required_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography', 'Gender']
            # missing_cols = [col for col in required_cols if col not in df.columns]
            # if missing_cols:
            #     return jsonify({'error': f'Missing columns in CSV: {", ".join(missing_cols)}'}), 400

            for _, row in df.iterrows():
                try:
                    # Convert row to dictionary, handling potential type issues
                    data_dict = row.to_dict()
                    # Ensure numeric types where appropriate, similar to single predict
                    for key in ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']:
                        if key in data_dict and data_dict[key] is not None:
                            try:
                                data_dict[key] = pd.to_numeric(data_dict[key])
                            except ValueError:
                                # Keep original if conversion fails, predict_churn might handle it or raise error
                                pass 
                    
                    prediction_result = predict_churn(data_dict)
                    # Add original data along with prediction
                    output_row = row.to_dict()
                    output_row['ChurnPrediction'] = prediction_result.get('churn_prediction', 'Error')
                    output_row['ChurnProbability'] = prediction_result.get('churn_probability', 'Error')
                    predictions.append(output_row)
                except Exception as e:
                    # If a single row fails, add error info for that row
                    error_row = row.to_dict()
                    error_row['ChurnPrediction'] = 'Error'
                    error_row['ChurnProbability'] = str(e)
                    predictions.append(error_row)
            
            output_df = pd.DataFrame(predictions)
            
            # Create a CSV in memory
            csv_buffer = io.StringIO()
            output_df.to_csv(csv_buffer, index=False)
            
            # Create a BytesIO buffer for sending the file
            mem_file = io.BytesIO()
            mem_file.write(csv_buffer.getvalue().encode('utf-8'))
            mem_file.seek(0)
            csv_buffer.close()
            
            return send_file(
                mem_file,
                mimetype='text/csv',
                as_attachment=True,
                download_name='batch_predictions.csv'
            )
        else:
            return jsonify({'error': 'Invalid file type. Please upload a .csv file'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# if __name__ == '__main__':
#     app.run(debug=True) # Commented out for production deployment
