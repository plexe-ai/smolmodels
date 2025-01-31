"""
This module provides the GoogleProvider class for interacting with Google's Generative AI API.

The GoogleProvider class extends the abstract Provider class and implements the
query method to send requests to the Google Generative AI API and handle responses.

Classes:
    - GoogleProvider: A provider for interacting with Google's Generative AI API.
"""

import logging
import os
from typing import Type

import google.generativeai as genai
from pydantic import BaseModel
from typing_extensions import TypedDict

from smolmodels.internal.common.providers.provider import Provider

logger = logging.getLogger(__name__)


class GoogleProvider(Provider):
    """
    A Provider implementation for interacting with Google's Generative AI API.
    """

    def __init__(self, api_key: str = None, model: str = "gemini-1.5-flash"):
        self.key = api_key or os.environ.get("GOOGLE_API_KEY", default=None)
        self.model = model or "gemini-1.5-flash"
        self.max_tokens = 4096

    def _query_impl(self, system_message: str, user_message: str, response_format: Type[BaseModel] = None) -> str:
        """
        Queries the Google Generative AI API with the given messages and returns the response.

        :param system_message: The system message to send.
        :param user_message: The user message to send.
        :param response_format: The format for the response. Defaults to None.
        :return: The content of the response from the OpenAI API.
        """
        if response_format is not None:
            generation_config = genai.GenerationConfig(
                max_output_tokens=self.max_tokens,
                response_mime_type="application/json",
                response_schema=TypedDict(  # convert Pydantic model to TypedDict
                    response_format.__name__, **{k: v.annotation for k, v in response_format.model_fields.items()}
                ),
            )
        else:
            generation_config = genai.GenerationConfig(max_output_tokens=self.max_tokens)

        llm = genai.GenerativeModel(
            model_name=self.model,
            generation_config=generation_config,
            system_instruction=system_message,
        )
        response = llm.generate_content(user_message)
        return response.text
