"""
Logging configuration for the medical imaging backend.
"""
import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional
from app.core.config import settings


def setup_logging() -> None:
    """Configure application logging."""
    
    # Create logs directory if it doesn't exist
    log_dir = Path(settings.LOG_FILE).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        filename=settings.LOG_FILE,
        maxBytes=settings.LOG_MAX_SIZE,
        backupCount=settings.LOG_BACKUP_COUNT,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler for development
    console_handler = logging.StreamHandler()
    if settings.DEBUG:
        console_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Configure specific loggers
    setup_specific_loggers()
    
    # Log startup message
    logging.info("Logging configuration completed")
    logging.info(f"Log level: {settings.LOG_LEVEL}")
    logging.info(f"Log file: {settings.LOG_FILE}")


def setup_specific_loggers() -> None:
    """Configure specific module loggers."""
    
    # SQLAlchemy logging
    sqlalchemy_logger = logging.getLogger('sqlalchemy')
    if settings.DEBUG:
        sqlalchemy_logger.setLevel(logging.INFO)
        sqlalchemy_logger.addHandler(logging.StreamHandler())
    else:
        sqlalchemy_logger.setLevel(logging.WARNING)
    
    # FastAPI logging
    fastapi_logger = logging.getLogger('fastapi')
    fastapi_logger.setLevel(logging.INFO)
    
    # Uvicorn logging
    uvicorn_logger = logging.getLogger('uvicorn')
    uvicorn_logger.setLevel(logging.INFO)
    
    # Celery logging
    celery_logger = logging.getLogger('celery')
    celery_logger.setLevel(logging.INFO)
    
    # Application specific loggers
    app_logger = logging.getLogger('app')
    app_logger.setLevel(logging.INFO)
    
    # Security logger (for audit trails)
    security_logger = logging.getLogger('app.security')
    security_logger.setLevel(logging.INFO)
    
    # ML inference logger
    ml_logger = logging.getLogger('app.ml')
    ml_logger.setLevel(logging.INFO)
    
    # API logger
    api_logger = logging.getLogger('app.api')
    api_logger.setLevel(logging.INFO)


class SecurityLogger:
    """Specialized logger for security events."""
    
    def __init__(self):
        self.logger = logging.getLogger('app.security')
        
        # Create security log file
        security_log_dir = Path("logs/security")
        security_log_dir.mkdir(parents=True, exist_ok=True)
        
        security_handler = logging.handlers.RotatingFileHandler(
            filename=security_log_dir / "security.log",
            maxBytes=settings.LOG_MAX_SIZE,
            backupCount=settings.LOG_BACKUP_COUNT,
            encoding='utf-8'
        )
        
        security_formatter = logging.Formatter(
            fmt='%(asctime)s - SECURITY - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        security_handler.setFormatter(security_formatter)
        self.logger.addHandler(security_handler)
    
    def log_login_attempt(self, email: str, ip_address: str, success: bool):
        """Log login attempt."""
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(f"LOGIN_ATTEMPT - {status} - {email} - {ip_address}")
    
    def log_logout(self, user_id: int, ip_address: str):
        """Log user logout."""
        self.logger.info(f"LOGOUT - User {user_id} - {ip_address}")
    
    def log_file_upload(self, user_id: int, filename: str, file_size: int, ip_address: str):
        """Log file upload."""
        self.logger.info(f"FILE_UPLOAD - User {user_id} - {filename} - {file_size} bytes - {ip_address}")
    
    def log_inference_request(self, user_id: int, model_type: str, image_id: int):
        """Log inference request."""
        self.logger.info(f"INFERENCE_REQUEST - User {user_id} - Model {model_type} - Image {image_id}")
    
    def log_data_access(self, user_id: int, resource_type: str, resource_id: int, action: str):
        """Log data access."""
        self.logger.info(f"DATA_ACCESS - User {user_id} - {resource_type} {resource_id} - {action}")
    
    def log_permission_denied(self, user_id: int, resource: str, reason: str):
        """Log permission denied."""
        self.logger.warning(f"PERMISSION_DENIED - User {user_id} - {resource} - {reason}")


class AuditLogger:
    """Logger for audit trails and compliance."""
    
    def __init__(self):
        self.logger = logging.getLogger('app.audit')
        
        # Create audit log file
        audit_log_dir = Path("logs/audit")
        audit_log_dir.mkdir(parents=True, exist_ok=True)
        
        audit_handler = logging.handlers.RotatingFileHandler(
            filename=audit_log_dir / "audit.log",
            maxBytes=settings.LOG_MAX_SIZE,
            backupCount=settings.LOG_BACKUP_COUNT,
            encoding='utf-8'
        )
        
        audit_formatter = logging.Formatter(
            fmt='%(asctime)s - AUDIT - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        audit_handler.setFormatter(audit_formatter)
        self.logger.addHandler(audit_handler)
    
    def log_data_change(self, user_id: int, table: str, record_id: int, action: str, old_values: dict, new_values: dict):
        """Log data changes for audit trail."""
        self.logger.info(f"DATA_CHANGE - User {user_id} - {table} {record_id} - {action} - Old: {old_values} - New: {new_values}")
    
    def log_system_event(self, event_type: str, description: str, details: dict):
        """Log system events."""
        self.logger.info(f"SYSTEM_EVENT - {event_type} - {description} - {details}")
    
    def log_compliance_event(self, event_type: str, user_id: Optional[int], description: str):
        """Log compliance-related events."""
        user_info = f"User {user_id}" if user_id else "System"
        self.logger.info(f"COMPLIANCE - {event_type} - {user_info} - {description}")


# Global instances
security_logger = SecurityLogger()
audit_logger = AuditLogger()


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


def log_function_entry(logger: logging.Logger, func_name: str, **kwargs):
    """Log function entry with parameters."""
    params = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.debug(f"Entering {func_name}({params})")


def log_function_exit(logger: logging.Logger, func_name: str, result=None):
    """Log function exit with result."""
    if result is not None:
        logger.debug(f"Exiting {func_name} with result: {result}")
    else:
        logger.debug(f"Exiting {func_name}")


def log_error(logger: logging.Logger, error: Exception, context: str = ""):
    """Log error with context."""
    context_str = f" in {context}" if context else ""
    logger.error(f"Error{context_str}: {error}", exc_info=True)


def log_performance_metric(logger: logging.Logger, operation: str, duration: float, **metrics):
    """Log performance metrics."""
    metric_str = ", ".join(f"{k}={v}" for k, v in metrics.items())
    logger.info(f"PERFORMANCE - {operation} took {duration:.3f}s - {metric_str}")


