# src/utils/logging_config.py
"""
Structured logging configuration for better observability
[Intent] Production-ready logging with structured output for monitoring
[DevOps] Integrates with logging aggregation systems (ELK, Splunk, etc.)
"""

import structlog
import logging
import sys
from typing import Any, Dict
from config.settings import get_config

config = get_config()

def configure_logging() -> None:
    """Configure structured logging for the application"""
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if not config.DEBUG else structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, config.LOG_LEVEL.upper()),
    )

def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a configured logger instance"""
    return structlog.get_logger(name)

# Application-specific loggers
app_logger = get_logger("app")
api_logger = get_logger("api") 
model_logger = get_logger("model")
metrics_logger = get_logger("metrics")
