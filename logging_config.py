import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

def setup_logging(log_level: str = "INFO", log_file: str = None):
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
    """
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file) if log_file else "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Default log file
    if not log_file:
        log_file = os.path.join(log_dir, f"research_chatbot_{datetime.now().strftime('%Y%m%d')}.log")
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            # Console handler
            logging.StreamHandler(),
            # File handler with rotation
            RotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
        ]
    )
    
    # Set specific logger levels
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('chromadb').setLevel(logging.WARNING)
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    
    logging.info("Logging configured successfully")
    
    return logging.getLogger(__name__)