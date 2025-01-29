"""
Application entry point for the data generation service.

The data generation service is an internal API that generates synthetic data that is meant to capture a particular
data distribution, either with data or without data (low-data regime). The service also exposes functionality for
validating the synthetic data against real data, if available.
"""

import logging
import warnings

from .config import config


# configure warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
# configure logging
logger = logging.getLogger()  # Get the root logger
logger.setLevel(logging.DEBUG)  # Set the root logger's level

# Configure handlers
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.WARNING)
stream_handler.stream.reconfigure(encoding="utf-8")  # Set UTF-8 encoding

file_handler = logging.FileHandler("smolmodels.log", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)

# Define a common formatter
formatter = logging.Formatter(config.FORMAT)
stream_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the root logger
logger.addHandler(stream_handler)
logger.addHandler(file_handler)
