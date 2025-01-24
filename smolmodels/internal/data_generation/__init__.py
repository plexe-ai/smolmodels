"""
Application entry point for the data generation service.

The data generation service is an internal API that generates synthetic data that is meant to capture a particular
data distribution, either with data or without data (low-data regime). The service also exposes functionality for
validating the synthetic data against real data, if available.
"""

import logging
import warnings

from .config import config
from .core.generation.combined import CombinedDataGenerator
from .core.generation.simple_llm import SimpleLLMDataGenerator
from .core.validation.eda import EdaDataValidator


# configure warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
# configure logging
logging.basicConfig(
    level=config.LEVEL, format=config.FORMAT, handlers=[logging.StreamHandler(), logging.FileHandler("app.log")]
)
logging.getLogger().handlers[0].setLevel(config.LEVEL)
logging.getLogger().handlers[1].setLevel(logging.DEBUG)


# initialise the generator depending on the config, then pass the config to it
generator = {"simple": SimpleLLMDataGenerator, "combined": CombinedDataGenerator}[config.GENERATOR](config)
# initialise the validator depending on the config
validator = {"eda": EdaDataValidator()}[config.VALIDATOR]
