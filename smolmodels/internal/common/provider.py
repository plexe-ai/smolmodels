"""
This module defines the base class for LLM providers and includes
logging and retry mechanisms for querying the providers.
"""

import textwrap
from typing import Type
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel
import logging
from litellm import completion

logger = logging.getLogger(__name__)


class Provider:
    """
    Base class for LiteLLM provider.
    """

    def __init__(self, model: str = None):
        default_model = "openai/gpt-4o-mini"
        self.model = model or default_model
        if "/" not in self.model:
            self.model = default_model
            logger.warning(f"Model name should be in the format 'provider/model', using default model: {default_model}")

    def query(
        self,
        system_message: str,
        user_message: str,
        response_format: Type[BaseModel] = None,
        retries: int = 3,
        backoff: bool = True,
    ) -> str:
        """
        Method to query the provider using litellm.completion.

        :param [str] system_message: The system message to send to the provider.
        :param [str] user_message: The user message to send to the provider.
        :param [Type[BaseModel]] response_format: A pydantic BaseModel class representing the response format.
        :param [int] retries: The number of times to retry the request. Defaults to 3.
        :param [bool] backoff: Whether to use exponential backoff when retrying. Defaults to True.
        :return [str]: The response from the provider.
        """
        self._log_request(system_message, user_message, self.__class__.__name__)

        messages = [{"role": "system", "content": system_message}, {"role": "user", "content": user_message}]
        try:
            if backoff:

                @retry(stop=stop_after_attempt(retries), wait=wait_exponential(multiplier=2))
                def call_with_backoff():
                    response = completion(model=self.model, messages=messages, response_format=response_format)
                    return response.choices[0].message.content

                r = call_with_backoff()
            else:
                response = completion(model=self.model, messages=messages, response_format=response_format)
                r = response.choices[0].message.content

            self._log_response(r, self.__class__.__name__)
            return r
        except Exception as e:
            self._log_error(e)
            raise e

    @staticmethod
    def _log_request(system_message: str, user_message: str, model):
        """
        Logs the request to the provider.

        :param [str] system_message: The system message to send to the provider.
        :param [str] user_message: The user message to send to the provider.
        """
        logger.debug(
            (
                f"Requesting chat completion from {model} with messages: "
                f"{textwrap.shorten(system_message.replace("\n", " "), 30)}, "
                f"{textwrap.shorten(user_message.replace("\n", " "), 30)}"
            )
        )

    @staticmethod
    def _log_response(response, model):
        """
        Logs the response from the provider.

        :param [str] response: The response from the provider.
        """
        logger.debug(f"Received completion from {model}: {textwrap.shorten(response, 30)}")

    @staticmethod
    def _log_error(error):
        """
        Logs the error from the provider.

        :param [str] error: The error from the provider.
        """
        logger.error(f"Error querying provider: {error}")
