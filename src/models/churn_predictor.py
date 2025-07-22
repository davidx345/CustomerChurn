# src/models/churn_predictor.py
"""
Enhanced churn prediction model with explainability and ethical guardrails
[Intent] Production-ready ML model with monitoring, validation, and transparency
[Security] Input validation and sanitization
[Ethics] Bias detection and overconfidence flagging
"""

import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import joblib
from src.utils.logging_config import model_logger
from src.utils.metrics import track_prediction
from config.settings import get_config

config = get_config()

class ChurnPredictor:
    """Enhanced customer churn predictor with explainability"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or config.MODEL_PATH
        self.model = None
        self.explainer = None
        self.feature_names = None
        self.model_metadata = {}
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the trained model and explainer"""
        try:
            model_bundle = joblib.load(self.model_path)
            self.model = model_bundle.get('model')
            self.explainer = model_bundle.get('explainer')
            self.feature_names = model_bundle.get('features', [])
            self.model_metadata = model_bundle.get('metadata', {})
            
            model_logger.info(
                "Model loaded successfully",
                model_path=self.model_path,
                model_type=str(type(self.model).__name__),
                feature_count=len(self.feature_names)
            )
        except Exception as e:
            model_logger.error("Failed to load model", error=str(e), model_path=self.model_path)
            raise
    
    def _validate_input(self, input_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate input data for required features and data types"""
        required_features = [
            'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 
            'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
        ]
        
        # Check for missing features
        missing_features = [feat for feat in required_features if feat not in input_data]
        if missing_features:
            return False, f"Missing required features: {', '.join(missing_features)}"
        
        # Validate data types and ranges
        try:
            # Numeric validations
            if not (300 <= float(input_data['CreditScore']) <= 850):
                return False, "CreditScore must be between 300 and 850"
            
            if not (18 <= int(input_data['Age']) <= 120):
                return False, "Age must be between 18 and 120"
            
            if not (0 <= int(input_data['Tenure']) <= 50):
                return False, "Tenure must be between 0 and 50 years"
            
            if float(input_data['Balance']) < 0:
                return False, "Balance cannot be negative"
            
            if not (1 <= int(input_data['NumOfProducts']) <= 10):
                return False, "NumOfProducts must be between 1 and 10"
            
            # Categorical validations
            if input_data['Geography'] not in ['France', 'Spain', 'Germany']:
                return False, "Geography must be France, Spain, or Germany"
            
            if input_data['Gender'] not in ['Male', 'Female']:
                return False, "Gender must be Male or Female"
            
            if int(input_data['HasCrCard']) not in [0, 1]:
                return False, "HasCrCard must be 0 or 1"
            
            if int(input_data['IsActiveMember']) not in [0, 1]:
                return False, "IsActiveMember must be 0 or 1"
            
        except (ValueError, TypeError) as e:
            return False, f"Invalid data type: {str(e)}"
        
        return True, None
    
    def _detect_bias(self, input_data: Dict[str, Any], probability: float) -> Dict[str, Any]:
        """Detect potential bias in predictions based on protected attributes"""
        bias_flags = {}
        
        # Age bias detection
        age = int(input_data['Age'])
        if age > 60 and probability > 0.7:
            bias_flags['age_bias'] = "High churn probability for older customer - verify fairness"
        
        # Gender bias detection (simplified)
        gender = input_data['Gender']
        if gender == 'Female' and probability > 0.8:
            bias_flags['gender_bias'] = "High churn probability for female customer - review for bias"
        
        # Geography bias
        geography = input_data['Geography']
        if geography in ['Spain', 'Germany'] and probability > 0.75:
            bias_flags['geography_bias'] = f"High churn probability for {geography} - check regional fairness"
        
        return bias_flags
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction with explainability and ethical guardrails
        
        Returns:
            Dict containing prediction, probability, feature_importance, transparency info, and bias flags
        """
        # Input validation
        is_valid, validation_error = self._validate_input(input_data)
        if not is_valid:
            model_logger.warning("Input validation failed", error=validation_error, input_data=input_data)
            return {
                'error': validation_error,
                'transparency': 'Input validation failed - prediction not generated',
                'harmfulness_flag': False
            }
        
        try:
            # Prepare input DataFrame
            df = pd.DataFrame([input_data])
            
            # Make prediction
            prediction = self.model.predict(df)[0]
            probability = self.model.predict_proba(df)[0][1]  # Probability of churn (class 1)
            
            # Track metrics
            track_prediction(prediction)
            
            # Calculate feature importance (SHAP if available)
            feature_importance = {}
            try:
                if self.explainer and config.ENABLE_SHAP_EXPLANATIONS:
                    shap_values = self.explainer.shap_values(df)
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1][0]  # Churn class
                    else:
                        shap_values = shap_values[0]
                    
                    feature_importance = dict(zip(df.columns, shap_values.astype(float)))
                else:
                    # Fallback to feature importances if available
                    if hasattr(self.model, 'feature_importances_'):
                        feature_importance = dict(zip(df.columns, self.model.feature_importances_))
            except Exception as e:
                model_logger.warning("Failed to calculate feature importance", error=str(e))
                feature_importance = {'explainability_error': str(e)}
            
            # Ethical guardrails
            harmfulness_flag = False
            bias_flags = {}
            
            if config.ENABLE_ETHICAL_GUARDRAILS:
                # Overconfidence detection
                if probability > 0.99 or probability < 0.01:
                    harmfulness_flag = True
                
                # Bias detection
                bias_flags = self._detect_bias(input_data, probability)
            
            # Transparency labeling
            transparency_parts = []
            transparency_parts.append(f"Model: {type(self.model).__name__}")
            transparency_parts.append(f"Version: {self.model_metadata.get('version', 'unknown')}")
            
            if config.ENABLE_SHAP_EXPLANATIONS:
                transparency_parts.append("SHAP explanations enabled")
            
            if harmfulness_flag:
                transparency_parts.append("Overconfidence detected")
            
            if bias_flags:
                transparency_parts.append("Bias flags raised")
            
            transparency = " | ".join(transparency_parts)
            
            result = {
                'prediction': int(prediction),
                'probability': float(probability),
                'feature_importance': feature_importance,
                'transparency': transparency,
                'harmfulness_flag': harmfulness_flag,
                'bias_flags': bias_flags,
                'model_version': self.model_metadata.get('version', 'unknown')
            }
            
            model_logger.info(
                "Prediction completed",
                prediction=prediction,
                probability=probability,
                harmfulness_flag=harmfulness_flag,
                bias_count=len(bias_flags)
            )
            
            return result
            
        except Exception as e:
            model_logger.error("Prediction failed", error=str(e), input_data=input_data)
            return {
                'error': f'Prediction failed: {str(e)}',
                'transparency': 'Internal error during prediction',
                'harmfulness_flag': False
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata and information"""
        return {
            'model_type': str(type(self.model).__name__) if self.model else 'Unknown',
            'feature_count': len(self.feature_names),
            'features': self.feature_names,
            'metadata': self.model_metadata,
            'explainer_available': self.explainer is not None,
            'model_path': self.model_path
        }
