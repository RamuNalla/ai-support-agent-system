import logging
import sys

def setup_logging():                    # Basic logging for the application.

    logger = logging.getLogger()        # Get the root logger
    logger.setLevel(logging.INFO)       # Set the minimum level for logging (warning messages of INFO< WARNING, ERROR and CRITICAL)

    console_handler = logging.StreamHandler(sys.stdout)     # Sends the log messages to console
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(                          # Format of log messages
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)

    if not logger.handlers:                                 # Prevents adding duplicate handlers if this function is called multiple times.
        logger.addHandler(console_handler)                 
        
    logging.getLogger("uvicorn").setLevel(logging.INFO)     # Set uvicorn loggers to INFO to avoid excessive output
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("uvicorn.error").setLevel(logging.ERROR)

    logging.info("Logging configured successfully.")

