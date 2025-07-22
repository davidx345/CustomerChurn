# config/settings.py
"""
Configuration management for Customer Churn Prediction
[Intent] Centralized configuration with environment-based overrides
[Security] No secrets in code, environment variable integration
[DevOps] Different configs for dev/staging/prod environments
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Base configuration class"""
    
    # Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # Model settings
    MODEL_PATH = os.getenv('MODEL_PATH', 'churn_model.pkl')
    MODEL_VERSION = os.getenv('MODEL_VERSION', '1.0.0')
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # API settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file upload
    API_RATE_LIMIT = os.getenv('API_RATE_LIMIT', '100 per hour')
    
    # Database (for future expansion)
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///churn.db')
    
    # Redis (for caching)
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    
    # Monitoring
    PROMETHEUS_METRICS_PORT = int(os.getenv('PROMETHEUS_METRICS_PORT', '8000'))
    ENABLE_METRICS = os.getenv('ENABLE_METRICS', 'True').lower() == 'true'
    
    # Feature flags
    ENABLE_SHAP_EXPLANATIONS = os.getenv('ENABLE_SHAP_EXPLANATIONS', 'True').lower() == 'true'
    ENABLE_ETHICAL_GUARDRAILS = os.getenv('ENABLE_ETHICAL_GUARDRAILS', 'True').lower() == 'true'

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    LOG_LEVEL = 'WARNING'

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

# Configuration mapping
config_by_name: Dict[str, Any] = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config() -> Config:
    """Get configuration based on environment"""
    env = os.getenv('FLASK_ENV', 'default')
    return config_by_name.get(env, DevelopmentConfig)()
