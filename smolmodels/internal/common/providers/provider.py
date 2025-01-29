import abc
import textwrap
from typing import Type

from pydantic import BaseModel


class Provider(abc.ABC):
    """
    Abstract base class for LLM providers.
    """

    @abc.abstractmethod
    def query(self, system_message: str, user_message: str, response_format: Type[BaseModel] = None) -> str:
        """
        Abstract method to query the provider.

        :param [str] system_message: The system message to send to the provider.
        :param [str] user_message: The user message to send to the provider.
        :param [BaseModel] response_format: The format of the response.
        :return [str]: The response from the provider.
        """
        pass

    @staticmethod
    def _log_request(system_message: str, user_message: str, model, logger):
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
    def _log_response(response, model, logger):
        """
        Logs the response from the provider.

        :param [str] response: The response from the provider.
        """
        logger.debug(f"Received completion from {model}: {textwrap.shorten(response, 30)}")
