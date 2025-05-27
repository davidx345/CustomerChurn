"""
predict_churn.py
Helper functions for input validation and churn prediction using the saved model.
"""
import joblib
import numpy as np
import pandas as pd
import shap

# Load model and preprocessors
model_bundle = joblib.load('churn_model.pkl')
model = model_bundle['model']
scaler = model_bundle['scaler']
le_gender = model_bundle['le_gender']
le_geo = model_bundle['le_geo']
features = model_bundle['features'] # Original feature names before encoding/scaling
# The training data used for the explainer ideally should be a sample of the data
# used to train the model, in its *preprocessed* form.
# For now, we'll create a placeholder or assume it might be loaded if available.
# If X_train_processed was saved during training, load it. Otherwise, this is a simplification.
# explainer = shap.KernelExplainer(model.predict_proba, X_train_processed_sample)
# For tree-based models, TreeExplainer is more efficient.
# We need to know the type of 'model' to choose the best explainer.
# Assuming it's a tree-based model like RandomForest or XGBoost for TreeExplainer.
# If it's a different model type, KernelExplainer might be more appropriate but slower.
# For demonstration, let's try to create a TreeExplainer.
# This might fail if the model is not a compatible tree-based model.
try:
    explainer = shap.TreeExplainer(model)
except Exception: # Fallback or error if not a tree model
    # A more robust solution would be to save the training data sample
    # and use KernelExplainer:
    # X_train_sample_scaled = pd.DataFrame(scaler.transform(X_train_sample[numerical_features]), columns=numerical_features)
    # X_train_sample_encoded = pd.concat([X_train_sample_scaled, pd.DataFrame(encoder.transform(X_train_sample[categorical_features]), columns=encoded_categorical_features)], axis=1)
    # explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_train_sample_encoded, 100)) # Using 100 samples
    # For now, we'll skip creating the explainer if TreeExplainer fails,
    # as we don't have access to the training data here.
    # In a real scenario, this needs to be handled by saving a sample of training data
    # or using a model-specific explainer.
    explainer = None 
    print("Warning: SHAP TreeExplainer initialization failed. Model explainability will be limited. Ensure the model is tree-based or provide a sample of training data for KernelExplainer.")


def get_feature_names_after_preprocessing():
    """
    Utility function to get feature names after one-hot encoding and scaling.
    This is important for mapping SHAP values correctly.
    Note: This is a simplified version. A more robust way is to save
    the column transformer or the exact feature names from the training script.
    """
    # This needs to match exactly how features were created during training
    # We assume 'features' contains the original column names
    # And le_gender, le_geo were used for 'Gender' and 'Geography'
    # Other features were scaled.
    
    # This is a common source of error if not perfectly aligned with training.
    # For simplicity, we'll use the 'features' list, but be aware that
    # one-hot encoding (if used beyond simple label encoding for Gender/Geo)
    # would expand the feature space. The current 'features' list seems to be
    # the original features before any transformation.
    # The SHAP explainer for scikit-learn models usually handles this if given the model and data.
    # However, our current `validate_input` returns a numpy array without column names.
    
    # Let's assume the 'features' list is the correct order for the model input
    # *after* label encoding but *before* scaling for numerical.
    # SHAP values are often returned for the features as seen by the model.
    return features


def validate_input(data):
    """
    Validate and preprocess input data for prediction.
    Returns a Pandas DataFrame ready for model prediction or raises ValueError.
    """
    required = set(features) # Assuming 'features' are the original column names expected by the model
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"Missing fields: {', '.join(missing)}")
    
    # Create a DataFrame from the input data, ensuring correct order via 'features'
    # This df_input will have original feature names and values.
    df_input_ordered = pd.DataFrame([data])[features]

    # Apply transformations (encoding, scaling) in the same way as training
    # This part needs to carefully mirror the preprocessing in train_model.py
    
    # Create a copy for transformation
    df_transformed = df_input_ordered.copy()

    # 1. Encode categorical features
    if 'Gender' in df_transformed.columns:
        df_transformed['Gender'] = le_gender.transform(df_transformed['Gender'])
    if 'Geography' in df_transformed.columns:
        df_transformed['Geography'] = le_geo.transform(df_transformed['Geography'])

    # 2. Scale features
    # The scaler was fit on data where categorical features were already encoded.
    # The `features` list from the bundle should represent the column names
    # of the data *after* encoding but *before* scaling, if the scaler is applied to all of them.
    # Or, they are the final feature names if the scaler was part of a pipeline that preserved names.
    # The current `train_model.py` scales all features in the `features` list *after* they have been label encoded.
    
    # So, df_transformed now has encoded categoricals and original numericals.
    # All columns in df_transformed (which are named by `features`) should be scaled.
    df_scaled_values = scaler.transform(df_transformed)
    
    # Create a new DataFrame with scaled values and original feature names
    # This is the DataFrame that the model expects.
    df_ready_for_model = pd.DataFrame(df_scaled_values, columns=features)
    
    return df_ready_for_model


def predict_churn(data):
    """
    Predict churn (Yes/No) and probability for a single customer dict.
    Returns: dict with 'churn_prediction', 'churn_probability', and 'shap_values'.
    """
    # validate_input now returns a DataFrame with feature names
    df_ready_for_model = validate_input(data)
    
    proba = model.predict_proba(df_ready_for_model)[0, 1] # Probability of churn (class 1)
    pred = model.predict(df_ready_for_model)[0] # 0 or 1

    shap_values_dict = None
    if explainer:
        try:
            # Pass the DataFrame directly to the explainer
            shap_values_instance = explainer.shap_values(df_ready_for_model)

            if isinstance(shap_values_instance, list) and len(shap_values_instance) == 2: # Binary classification
                shap_values_for_churn = shap_values_instance[1][0] # SHAP values for the positive class (churn)
            else: # Single output or different structure (e.g. some SHAP explainers return a single array for binary)
                # This might be (1, n_features) or directly (n_features,)
                if isinstance(shap_values_instance, np.ndarray) and shap_values_instance.ndim == 2 and shap_values_instance.shape[0] == 1:
                    shap_values_for_churn = shap_values_instance[0]
                else:
                    shap_values_for_churn = shap_values_instance # Assumes it's already (n_features,)

            # Ensure shap_values_for_churn is a 1D array (e.g., convert from (n_features, 1) to (n_features,))
            if hasattr(shap_values_for_churn, 'ndim') and shap_values_for_churn.ndim > 1:
                shap_values_for_churn = shap_values_for_churn.flatten()

            # Ensure SHAP values are converted to standard Python floats
            shap_values_for_churn_native = [float(val) for val in shap_values_for_churn]
            
            # Feature names are directly from df_ready_for_model.columns
            shap_values_dict = dict(zip(df_ready_for_model.columns, shap_values_for_churn_native))
        except Exception as e:
            print(f"Error calculating SHAP values: {e}")
            shap_values_dict = {"error": "Could not calculate SHAP values."}
            
    return {
        'churn_prediction': 1 if pred == 1 else 0, # Return 0 or 1 consistently
        'churn_probability': float(proba), # Ensure probability is float
        'shap_values': shap_values_dict
    }

# Example of how X_train_processed_sample might be created and saved during training:
# from sklearn.model_selection import train_test_split
# ... load your full dataset ...
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# ... preprocess X_train to X_train_processed ...
# X_train_processed_sample = shap.sample(X_train_processed, 100) # Or pd.DataFrame(X_train_processed).sample(100)
# joblib.dump({..., 'X_train_processed_sample': X_train_processed_sample}, 'churn_model.pkl')
# Then load it here:
# X_train_processed_sample = model_bundle.get('X_train_processed_sample')
# if X_train_processed_sample is not None:
#     explainer = shap.KernelExplainer(model.predict_proba, X_train_processed_sample)
# else:
#     print("Warning: X_train_processed_sample not found in model bundle. SHAP KernelExplainer cannot be initialized.")
#     explainer = None
