import abc


class Provider(abc.ABC):
    """
    Abstract base class for LLM providers.
    """

    @abc.abstractmethod
    def query(self, system_message: str, user_message: str, response_format=None) -> str:
        """
        Abstract method to query the provider.

        :param [str] system_message: The system message to send to the provider.
        :param [str] user_message: The user message to send to the provider.
        :param [str] response_format: The format of the response.
        :return [str]: The response from the provider.
        """
        pass
