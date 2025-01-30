"""
This module defines the abstract base class for LLM providers and includes
logging and retry mechanisms for querying the providers.
"""

import abc
import textwrap
from typing import Type
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)


class Provider(abc.ABC):
    """
    Abstract base class for LLM providers.
    """

    def query(
        self,
        system_message: str,
        user_message: str,
        response_format: Type[BaseModel] = None,
        retries: int = 3,
        backoff: bool = True,
    ) -> str:
        """
        Abstract method to query the provider.

        :param [str] system_message: The system message to send to the provider.
        :param [str] user_message: The user message to send to the provider.
        :param [Type[BaseModel]] response_format: A pydantic BaseModel class representing the response format.
        :param [int] retries: The number of times to retry the request. Defaults to 3.
        :param [bool] backoff: Whether to use exponential backoff when retrying. Defaults to True.
        :return [str]: The response from the provider.
        """
        self._log_request(system_message, user_message, self.__class__.__name__)

        try:
            if backoff:

                @retry(stop=stop_after_attempt(retries), wait=wait_exponential(multiplier=2))
                def call_with_backoff():
                    return self._query_impl(system_message, user_message, response_format)

                r = call_with_backoff()
            else:
                r = self._query_impl(system_message, user_message, response_format)
            self._log_response(r, self.__class__.__name__)
            return r
        except Exception as e:
            self._log_error(e)
            raise e

    @abc.abstractmethod
    def _query_impl(self, system_message: str, user_message: str, response_format: Type[BaseModel] = None) -> str:
        """
        Abstract method to implement the query logic. This method is called by the provider
        class' query method.

        :param [str] system_message: The system message to send to the provider.
        :param [str] user_message: The user message to send to the provider.
        :param [Type[BaseModel]] response_format: A pydantic BaseModel class representing the response format.
        :return [str]: The response from the provider.
        """
        pass

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
