"""
This module provides the DeepSeekProvider class for interacting with DeepSeek's API.

The DeepSeekProvider class extends the abstract Provider class and implements the
query method to send requests to the DeepSeek API and handle responses.

Classes:
    - DeepSeekProvider: A provider for interacting with DeepSeek's API.
"""

import logging
import os
from typing import Type

import openai
from pydantic import BaseModel

from smolmodels.internal.common.providers.provider import Provider

logger = logging.getLogger(__name__)


class DeepSeekProvider(Provider):
    """
    A Provider implementation for interacting with DeepSeek's API.
    """

    def __init__(self, api_key: str = None, model: str = "deepseek-chat"):
        self.key = api_key or os.environ.get("DEEPSEEK_API_KEY", default=None)
        self.model = model
        self.client = openai.OpenAI(api_key=self.key, base_url="https://api.deepseek.com")

    def query(self, system_message: str, user_message: str, response_format: Type[BaseModel] = None) -> str:
        """
        Queries the DeepSeek API with the given messages and returns the response.

        :param system_message: The system message to send.
        :param user_message: The user message to send.
        :param response_format: The format for the response. Defaults to None.
        :return: The content of the response from the OpenAI API.
        """
        self._log_request(system_message, user_message, self.model, logger)

        if response_format is not None:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                response_format=response_format,
            )
            content = response.choices[0].message.content
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
            )
            content = response.choices[0].message.content

        self._log_response(content, self.model, logger)

        return content
