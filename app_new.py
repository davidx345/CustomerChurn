# app.py - Refactored main application
"""
Refactored Flask application with improved structure and production features
[Intent] Production-ready Flask app with proper configuration and monitoring
[DevOps] Health checks, metrics, and structured logging
[Security] Enhanced error handling and input validation
"""

import os
import sys

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from flask import Flask, render_template
from src.api.routes import api_bp
from src.utils.logging_config import configure_logging, app_logger
from config.settings import get_config

# Configure logging first
configure_logging()

def create_app(config_name: str = None) -> Flask:
    """Application factory pattern"""
    app = Flask(__name__)
    
    # Load configuration
    config = get_config()
    app.config.from_object(config)
    
    # Register blueprints
    app.register_blueprint(api_bp)
    
    # Main route for frontend
    @app.route('/')
    def index():
        return render_template('index.html')
    
    # Global error handlers
    @app.errorhandler(404)
    def not_found(error):
        app_logger.warning("404 error", path=request.path if 'request' in globals() else 'unknown')
        return {'error': 'Endpoint not found', 'code': 'NOT_FOUND'}, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        app_logger.error("500 error", error=str(error))
        return {'error': 'Internal server error', 'code': 'INTERNAL_ERROR'}, 500
    
    @app.before_first_request
    def startup():
        app_logger.info("Application starting up", 
                       config_name=config.__class__.__name__,
                       debug=app.config['DEBUG'])
    
    return app

# Create the app
app = create_app()

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    
    app_logger.info("Starting development server", port=port, debug=debug)
    app.run(host='0.0.0.0', port=port, debug=debug)
