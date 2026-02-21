# src/core/logging_config.py
import logging
import sys
from pathlib import Path
from src.config.config import LOGGER_PATH


def setup_logging():
    log_dir = Path(LOGGER_PATH)
    log_dir.mkdir(exist_ok=True)

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Root Logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Console Handler with UTF-8 support
    # Note: On Windows, sys.stdout might still struggle if the terminal isn't UTF-8,
    # but forcing it here helps prevent the crash.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))

    # File Handler - EXPLICITLY set encoding to utf-8
    file_handler = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(log_format))

    # Clear existing handlers to avoid duplicates during reloads
    root_logger.handlers = []
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

# Initialize once at the module level
setup_logging()
logger = logging.getLogger("UniversityRAG")