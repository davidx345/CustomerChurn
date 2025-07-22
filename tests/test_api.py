# tests/test_api.py
"""
API endpoint tests for comprehensive coverage
[Intent] Ensure API reliability and proper error handling
[DevOps] Integration testing for CI/CD pipeline
"""

import pytest
import json
import io
import sys
import os
from unittest.mock import Mock, patch

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from app_new import create_app

class TestAPI:
    """Test suite for API endpoints"""
    
    @pytest.fixture
    def app(self):
        """Create test app"""
        app = create_app('testing')
        app.config['TESTING'] = True
        return app
    
    @pytest.fixture
    def client(self, app):
        """Create test client"""
        return app.test_client()
    
    @pytest.fixture
    def sample_customer_data(self):
        """Sample customer data for testing"""
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
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get('/health')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'service' in data
        assert 'version' in data
        assert 'timestamp' in data
    
    def test_predict_success(self, client, sample_customer_data):
        """Test successful prediction"""
        with patch('src.models.churn_predictor.ChurnPredictor') as mock_predictor_class:
            # Mock predictor instance
            mock_predictor = Mock()
            mock_predictor.predict.return_value = {
                'prediction': 0,
                'probability': 0.3,
                'feature_importance': {'Age': 0.2, 'Balance': 0.15},
                'transparency': 'Model: RandomForest | Version: 1.0.0',
                'harmfulness_flag': False,
                'bias_flags': {},
                'model_version': '1.0.0'
            }
            mock_predictor_class.return_value = mock_predictor
            
            response = client.post('/predict', 
                                 data=json.dumps(sample_customer_data),
                                 content_type='application/json')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'prediction' in data
            assert 'probability' in data
            assert 'api_version' in data
    
    def test_predict_empty_request(self, client):
        """Test prediction with empty request"""
        response = client.post('/predict', 
                             data='',
                             content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['code'] == 'EMPTY_REQUEST'
    
    def test_predict_invalid_format(self, client):
        """Test prediction with invalid format"""
        response = client.post('/predict', 
                             data=json.dumps([1, 2, 3]),  # Array instead of object
                             content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['code'] == 'INVALID_FORMAT'
    
    def test_predict_validation_error(self, client):
        """Test prediction with validation error"""
        with patch('src.models.churn_predictor.ChurnPredictor') as mock_predictor_class:
            mock_predictor = Mock()
            mock_predictor.predict.return_value = {
                'error': 'Missing required features: Age',
                'transparency': 'Input validation failed'
            }
            mock_predictor_class.return_value = mock_predictor
            
            invalid_data = {'CreditScore': 650}  # Missing required fields
            response = client.post('/predict', 
                                 data=json.dumps(invalid_data),
                                 content_type='application/json')
            
            assert response.status_code == 400
            data = json.loads(response.data)
            assert 'error' in data
    
    def test_batch_predict_no_file(self, client):
        """Test batch prediction without file"""
        response = client.post('/batch_predict')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['code'] == 'NO_FILE'
    
    def test_batch_predict_empty_filename(self, client):
        """Test batch prediction with empty filename"""
        response = client.post('/batch_predict', 
                             data={'csvFile': (io.BytesIO(b''), '')})
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['code'] == 'EMPTY_FILENAME'
    
    def test_batch_predict_invalid_file_type(self, client):
        """Test batch prediction with invalid file type"""
        response = client.post('/batch_predict', 
                             data={'csvFile': (io.BytesIO(b'data'), 'test.txt')})
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['code'] == 'INVALID_FILE_TYPE'
    
    def test_batch_predict_success(self, client):
        """Test successful batch prediction"""
        # Create sample CSV content
        csv_content = '''CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary
650,France,Male,35,5,50000,2,1,1,75000
700,Spain,Female,40,3,60000,1,0,1,80000'''
        
        with patch('src.models.churn_predictor.ChurnPredictor') as mock_predictor_class:
            mock_predictor = Mock()
            mock_predictor.predict.return_value = {
                'prediction': 0,
                'probability': 0.3,
                'harmfulness_flag': False,
                'bias_flags': {}
            }
            mock_predictor_class.return_value = mock_predictor
            
            response = client.post('/batch_predict', 
                                 data={'csvFile': (io.BytesIO(csv_content.encode()), 'test.csv')})
            
            assert response.status_code == 200
            assert response.headers['Content-Type'] == 'text/csv; charset=utf-8'
    
    def test_feature_importance(self, client):
        """Test feature importance endpoint"""
        with patch('src.models.churn_predictor.ChurnPredictor') as mock_predictor_class:
            mock_predictor = Mock()
            mock_predictor.model = Mock()
            mock_predictor.model.feature_importances_ = [0.1, 0.2, 0.15, 0.05, 0.1, 0.15, 0.1, 0.05, 0.05, 0.05]
            mock_predictor.get_model_info.return_value = {
                'features': ['CreditScore', 'Age', 'Balance', 'Tenure', 'Geography', 'Gender', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'],
                'model_type': 'RandomForestClassifier'
            }
            mock_predictor_class.return_value = mock_predictor
            
            response = client.get('/api/feature_importance')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'features' in data
            assert 'importances' in data
            assert 'model_type' in data
    
    def test_model_info(self, client):
        """Test model info endpoint"""
        with patch('src.models.churn_predictor.ChurnPredictor') as mock_predictor_class:
            mock_predictor = Mock()
            mock_predictor.get_model_info.return_value = {
                'model_type': 'RandomForestClassifier',
                'feature_count': 10,
                'features': ['CreditScore', 'Age'],
                'metadata': {'version': '1.0.0'},
                'explainer_available': True,
                'model_path': 'churn_model.pkl'
            }
            mock_predictor_class.return_value = mock_predictor
            
            response = client.get('/api/model_info')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['model_type'] == 'RandomForestClassifier'
            assert data['feature_count'] == 10
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint"""
        response = client.get('/metrics')
        
        assert response.status_code == 200
        assert response.headers['Content-Type'] == 'text/plain; charset=utf-8'
        # Should return Prometheus-formatted metrics
        assert b'# ' in response.data  # Prometheus comments start with #

if __name__ == '__main__':
    pytest.main([__file__])
