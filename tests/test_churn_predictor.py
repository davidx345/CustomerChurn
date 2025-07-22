# tests/test_churn_predictor.py
"""
Unit tests for churn predictor with comprehensive coverage
[Intent] Ensure model reliability and catch regressions
[DevOps] Automated testing in CI/CD pipeline
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.models.churn_predictor import ChurnPredictor

class TestChurnPredictor:
    """Test suite for ChurnPredictor class"""
    
    @pytest.fixture
    def sample_input(self):
        """Sample valid input data"""
        return {
            'CreditScore': 650,
            'Geography': 'France',
            'Gender': 'Male',
            'Age': 35,
            'Tenure': 5,
            'Balance': 50000.0,
            'NumOfProducts': 2,
            'HasCrCard': 1,
            'IsActiveMember': 1,
            'EstimatedSalary': 75000.0
        }
    
    @pytest.fixture
    def mock_predictor(self):
        """Mock predictor with fake model"""
        with patch('src.models.churn_predictor.joblib.load') as mock_load:
            # Mock model
            mock_model = Mock()
            mock_model.predict.return_value = np.array([0])
            mock_model.predict_proba.return_value = np.array([[0.7, 0.3]])
            mock_model.feature_importances_ = np.array([0.1, 0.2, 0.15, 0.05, 0.1, 0.15, 0.1, 0.05, 0.05, 0.05])
            
            mock_load.return_value = {
                'model': mock_model,
                'explainer': None,
                'features': ['CreditScore', 'Geography_France', 'Geography_Germany', 'Gender_Male', 
                           'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember'],
                'metadata': {'version': '1.0.0'}
            }
            
            predictor = ChurnPredictor('test_model.pkl')
            return predictor
    
    def test_validate_input_valid(self, mock_predictor, sample_input):
        """Test input validation with valid data"""
        is_valid, error = mock_predictor._validate_input(sample_input)
        assert is_valid is True
        assert error is None
    
    def test_validate_input_missing_features(self, mock_predictor):
        """Test input validation with missing features"""
        incomplete_input = {'CreditScore': 650, 'Age': 35}
        is_valid, error = mock_predictor._validate_input(incomplete_input)
        assert is_valid is False
        assert 'Missing required features' in error
    
    def test_validate_input_invalid_ranges(self, mock_predictor, sample_input):
        """Test input validation with invalid ranges"""
        # Test invalid credit score
        invalid_input = sample_input.copy()
        invalid_input['CreditScore'] = 200  # Too low
        is_valid, error = mock_predictor._validate_input(invalid_input)
        assert is_valid is False
        assert 'CreditScore must be between' in error
        
        # Test invalid age
        invalid_input = sample_input.copy()
        invalid_input['Age'] = 150  # Too high
        is_valid, error = mock_predictor._validate_input(invalid_input)
        assert is_valid is False
        assert 'Age must be between' in error
    
    def test_validate_input_invalid_categories(self, mock_predictor, sample_input):
        """Test input validation with invalid categorical values"""
        # Test invalid geography
        invalid_input = sample_input.copy()
        invalid_input['Geography'] = 'Canada'  # Not in allowed values
        is_valid, error = mock_predictor._validate_input(invalid_input)
        assert is_valid is False
        assert 'Geography must be' in error
        
        # Test invalid gender
        invalid_input = sample_input.copy()
        invalid_input['Gender'] = 'Other'  # Not in allowed values
        is_valid, error = mock_predictor._validate_input(invalid_input)
        assert is_valid is False
        assert 'Gender must be' in error
    
    def test_predict_success(self, mock_predictor, sample_input):
        """Test successful prediction"""
        result = mock_predictor.predict(sample_input)
        
        assert 'error' not in result
        assert 'prediction' in result
        assert 'probability' in result
        assert 'feature_importance' in result
        assert 'transparency' in result
        assert 'harmfulness_flag' in result
        assert 'bias_flags' in result
        
        assert isinstance(result['prediction'], int)
        assert isinstance(result['probability'], float)
        assert 0 <= result['probability'] <= 1
    
    def test_predict_validation_error(self, mock_predictor):
        """Test prediction with invalid input"""
        invalid_input = {'CreditScore': 650}  # Missing required features
        result = mock_predictor.predict(invalid_input)
        
        assert 'error' in result
        assert 'Missing required features' in result['error']
        assert result['transparency'] == 'Input validation failed - prediction not generated'
    
    def test_bias_detection_age(self, mock_predictor, sample_input):
        """Test age bias detection"""
        # Mock high probability prediction for older customer
        mock_predictor.model.predict_proba.return_value = np.array([[0.2, 0.8]])
        
        older_customer = sample_input.copy()
        older_customer['Age'] = 65
        
        bias_flags = mock_predictor._detect_bias(older_customer, 0.8)
        assert 'age_bias' in bias_flags
    
    def test_bias_detection_gender(self, mock_predictor, sample_input):
        """Test gender bias detection"""
        # Mock very high probability prediction for female customer
        mock_predictor.model.predict_proba.return_value = np.array([[0.1, 0.9]])
        
        female_customer = sample_input.copy()
        female_customer['Gender'] = 'Female'
        
        bias_flags = mock_predictor._detect_bias(female_customer, 0.9)
        assert 'gender_bias' in bias_flags
    
    def test_overconfidence_detection(self, mock_predictor, sample_input):
        """Test overconfidence detection"""
        # Mock overconfident prediction
        mock_predictor.model.predict_proba.return_value = np.array([[0.001, 0.999]])
        
        result = mock_predictor.predict(sample_input)
        assert result['harmfulness_flag'] is True
        assert 'Overconfidence detected' in result['transparency']
    
    def test_get_model_info(self, mock_predictor):
        """Test model info retrieval"""
        info = mock_predictor.get_model_info()
        
        assert 'model_type' in info
        assert 'feature_count' in info
        assert 'features' in info
        assert 'metadata' in info
        assert 'explainer_available' in info
        assert 'model_path' in info
        
        assert info['feature_count'] == 10
        assert info['metadata']['version'] == '1.0.0'

if __name__ == '__main__':
    pytest.main([__file__])
