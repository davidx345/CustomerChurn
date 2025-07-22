# src/utils/metrics.py
"""
Application metrics collection for monitoring and observability
[Intent] Production metrics for performance monitoring and alerting
[DevOps] Prometheus-compatible metrics for Grafana dashboards
"""

import time
from typing import Dict, Any, Optional
from functools import wraps
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from config.settings import get_config

config = get_config()

# Prometheus metrics
PREDICTION_REQUESTS = Counter(
    'churn_prediction_requests_total',
    'Total number of prediction requests',
    ['endpoint', 'status']
)

PREDICTION_LATENCY = Histogram(
    'churn_prediction_duration_seconds',
    'Time spent on prediction requests',
    ['endpoint']
)

MODEL_PREDICTIONS = Counter(
    'churn_model_predictions_total',
    'Total number of model predictions',
    ['prediction_class']
)

ACTIVE_REQUESTS = Gauge(
    'churn_active_requests',
    'Number of active requests being processed'
)

BATCH_UPLOAD_SIZE = Histogram(
    'churn_batch_upload_size',
    'Size of batch upload files in bytes'
)

def track_request_metrics(endpoint: str):
    """Decorator to track request metrics"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            ACTIVE_REQUESTS.inc()
            
            try:
                result = func(*args, **kwargs)
                PREDICTION_REQUESTS.labels(endpoint=endpoint, status='success').inc()
                return result
            except Exception as e:
                PREDICTION_REQUESTS.labels(endpoint=endpoint, status='error').inc()
                raise
            finally:
                ACTIVE_REQUESTS.dec()
                PREDICTION_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)
        
        return wrapper
    return decorator

def track_prediction(prediction_class: int):
    """Track individual prediction results"""
    class_name = 'churn' if prediction_class == 1 else 'no_churn'
    MODEL_PREDICTIONS.labels(prediction_class=class_name).inc()

def track_batch_size(file_size: int):
    """Track batch upload file sizes"""
    BATCH_UPLOAD_SIZE.observe(file_size)

def get_metrics() -> str:
    """Get Prometheus metrics in text format"""
    return generate_latest().decode('utf-8')
