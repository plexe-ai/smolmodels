"""
This module provides the AnthropicProvider class for interacting with Anthropic's API.

The AnthropicProvider class extends the abstract Provider class and implements the
query method to send requests to the Anthropic API and handle responses.

Classes:
    - AnthropicProvider: A provider for interacting with Anthropic's API.
"""

import logging
import os
from typing import Type

import anthropic
import instructor
from pydantic import BaseModel

from smolmodels.internal.common.providers.provider import Provider

logger = logging.getLogger(__name__)


class AnthropicProvider(Provider):
    """
    A Provider implementation for interacting with Anthropic's API.
    """

    def __init__(self, api_key: str = None, model: str = "claude-3-haiku-20240307"):
        self.key = api_key or os.environ.get("ANTHROPIC_API_KEY", default=None)
        self.model = model or "claude-3-haiku-20240307"
        self.max_tokens = 4096
        self.client = instructor.from_anthropic(anthropic.Anthropic(api_key=self.key))

    def query(self, system_message: str, user_message: str, response_format: Type[BaseModel] = None) -> str:
        """
        Queries the Anthropic API with the given messages and returns the response.

        :param system_message: The system message to send.
        :param user_message: The user message to send.
        :param response_format: The format for the response. Defaults to None.
        :return: The content of the response from the OpenAI API.
        """
        self._log_request(system_message, user_message, self.model, logger)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            response_model=response_format,
        )
        if response_format is None:
            content = str(response.content)
        else:
            content = str(response.model_dump_json(indent=4))

        self._log_response(content, self.model, logger)

        return str(content)
