# src/utils/logging_config.py

import logging
import sys
import os
from logging.handlers import RotatingFileHandler

# --- Configuration ---
LOG_LEVEL = logging.INFO  # Default level (INFO, DEBUG, WARNING, ERROR, CRITICAL)
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# File Logging (Optional)
ENABLE_FILE_LOGGING = False # Set to True to enable logging to a file
LOG_FILE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs') # Store logs in project root/logs
LOG_FILE_NAME = 'ai_tutor_bot.log'
LOG_FILE_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
LOG_FILE_BACKUP_COUNT = 5 # Keep 5 backup log files

def setup_logging(level=LOG_LEVEL):
    """
    Configures the root logger for the application.

    Args:
        level: The desired logging level (e.g., logging.INFO, logging.DEBUG).
    """
    # Get the root logger
    root_logger = logging.getLogger()

    # Avoid adding handlers multiple times if called repeatedly (e.g., in Streamlit reruns)
    if root_logger.hasHandlers():
        # Optional: Check if configuration is already as desired, or simply return
        # print("Logger already configured.") # Debug print
        return

    root_logger.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # --- Console Handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    logging.info("Console logging configured.") # Use logging here, not print

    # --- File Handler (Optional) ---
    if ENABLE_FILE_LOGGING:
        try:
            # Ensure log directory exists
            os.makedirs(LOG_FILE_DIR, exist_ok=True)
            log_file_path = os.path.join(LOG_FILE_DIR, LOG_FILE_NAME)

            # Use RotatingFileHandler for log rotation
            file_handler = RotatingFileHandler(
                log_file_path,
                maxBytes=LOG_FILE_MAX_BYTES,
                backupCount=LOG_FILE_BACKUP_COUNT,
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            logging.info(f"File logging configured. Log file: {log_file_path}")
        except Exception as e:
            # Log error about file logging setup to console
            logging.error(f"Failed to configure file logging to {LOG_FILE_DIR}: {e}", exc_info=True)

    # Set higher levels for noisy libraries if needed
    logging.getLogger("httpx").setLevel(logging.WARNING) # Reduce noise from http clients
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb.telemetry.posthog").setLevel(logging.WARNING) # Chroma telemetry
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logging.info(f"Root logger configured with level {logging.getLevelName(level)}.")

# Example of how to use it in other files:
# At the top of app.py, ingest.py, rag.py etc.:
#
# import logging
# from src.utils.logging_config import setup_logging
#
# setup_logging() # Call setup once
# logger = logging.getLogger(__name__) # Get logger for the current module
#
# # Now use logger as usual:
# logger.info("This is an info message from my module.")
# logger.debug("This is a debug message.")


# --- Main block for testing the logging setup ---
if __name__ == "__main__":
    print("Testing logging setup...")
    setup_logging(level=logging.DEBUG) # Test with DEBUG level

    # Get loggers for different hypothetical modules
    logger_main = logging.getLogger(__name__)
    logger_ingest = logging.getLogger("src.ingestion.ingest")
    logger_ui = logging.getLogger("src.ui.app")

    # Log messages at different levels
    logger_main.debug("This is a debug message from the main test block.")
    logger_main.info("This is an info message.")
    logger_main.warning("This is a warning message.")
    logger_main.error("This is an error message.")
    logger_main.critical("This is a critical message.")

    logger_ingest.info("Simulating an info message from the ingestion module.")
    logger_ui.debug("Simulating a debug message from the UI module.")

    try:
        1 / 0
    except ZeroDivisionError:
        logger_main.exception("Caught an exception!") # Logs error + traceback

    print("Logging setup test complete. Check console output" + (" and logs/" + LOG_FILE_NAME if ENABLE_FILE_LOGGING else "."))