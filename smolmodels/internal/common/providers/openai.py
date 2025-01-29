"""
This module provides the OpenAIProvider class for interacting with OpenAI's API.

The OpenAIProvider class extends the abstract Provider class and implements the
query method to send requests to the OpenAI API and handle responses.

Classes:
    - OpenAIProvider: A provider for interacting with OpenAI's API.
"""

import logging
import os

import openai
from pydantic import BaseModel
from typing import Type

from smolmodels.internal.common.providers.provider import Provider

logger = logging.getLogger(__name__)


class OpenAIProvider(Provider):
    """
    A Provider implementation for interacting with OpenAI's API.
    """

    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        self.key = api_key or os.environ.get("OPENAI_API_KEY", default=None)
        self.model = model or "gpt-4o-mini"
        self.max_tokens = 16000
        self.client = openai.OpenAI(api_key=self.key)

    def query(self, system_message: str, user_message: str, response_format: Type[BaseModel] = None) -> str:
        """
        Queries the OpenAI API with the given messages and returns the response.

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
                max_tokens=self.max_tokens,
            )
            content = response.choices[0].message.content
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=self.max_tokens,
            )
            content = response.choices[0].message.content

        self._log_response(content, self.model, logger)

        return content
