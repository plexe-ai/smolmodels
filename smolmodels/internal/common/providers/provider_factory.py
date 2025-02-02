"""
Module for creating and managing LLM providers using LiteLLM.

This module defines the `ProviderFactory` class, which is responsible for creating instances
of LiteLLM providers. LiteLLM provides a unified interface to multiple LLM providers including
OpenAI, Anthropic, Google, and others.

Classes:
    ProviderFactory: Factory class for creating LLM providers.

Usage example:
    factory = ProviderFactory()
    provider = factory.create("openai:gpt-4o-2024-08-06")
    response = provider.query(system_message="Hello", user_message="How are you?")
"""

from typing import Dict, Callable
from litellm import completion
from smolmodels.internal.common.providers.provider import Provider


class LiteLLMProvider(Provider):
    """
    Provider implementation that uses LiteLLM to interact with various LLM providers.
    """

    def __init__(self, model: str = None):
        self.model = model or "openai/gpt-4o-mini"

    def _query_impl(self, system_message: str, user_message: str, response_format=None) -> str:
        """
        Implementation of the query method using LiteLLM.

        :param system_message: The system message to send
        :param user_message: The user message to send
        :param response_format: Optional response format (not used in LiteLLM)
        :return: The response from the LLM
        """
        messages = [{"role": "system", "content": system_message}, {"role": "user", "content": user_message}]

        response = completion(model=self.model, messages=messages, response_format=response_format)
        return response.choices[0].message.content


class ProviderFactory:
    """
    Factory class for LLM providers using LiteLLM.
    """

    providers_map: Dict[str | None, Callable[[str | None], Provider]] = {
        None: lambda model: LiteLLMProvider(model="openai/gpt-4o-mini")
    }

    @staticmethod
    def create(provider_name: str | None = None) -> Provider:
        """
        Creates a provider based on the provider name. The provider name is expected to be in
        the format 'provider:model', where 'provider' is the name of the provider and 'model' is
        the name of the model. Valid input formats are 'provider:model' or None.

        :param provider_name: The name of the provider and model to use, in format 'provider:model'.
        :return: The provider.
        :raises TypeError: If provider_name is not a string or None
        :raises ValueError: If provider_name format is invalid
        """
        # Handle None case
        if provider_name is None:
            return LiteLLMProvider()

        # Type checking
        if not isinstance(provider_name, str):
            raise TypeError("Provider name must be a string or None")

        # Handle empty string
        if not provider_name.strip():
            return LiteLLMProvider()

        # Check for whitespace
        if provider_name != provider_name.strip():
            raise ValueError(f"Provider name '{provider_name}' contains leading or trailing whitespace")

        # Handle provider:model format
        if ":" in provider_name:
            parts = provider_name.split(":")
            if len(parts) > 2:
                raise ValueError(f"Invalid format: '{provider_name}'. Use 'provider:model'")
            if not parts[0] or not parts[1]:
                raise ValueError(f"Invalid format: '{provider_name}'. Provider and model cannot be empty")

            provider, model = parts
            return LiteLLMProvider(model=f"{provider}/{model}")

        # If only model is provided, assume it's a full model path
        return LiteLLMProvider(model=provider_name)
