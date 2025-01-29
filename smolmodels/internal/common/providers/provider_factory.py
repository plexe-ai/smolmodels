"""
Module for creating and managing LLM providers.

This module defines the `ProviderFactory` class, which is responsible for creating instances of different
LLM providers based on the specified provider name and model. The supported providers include OpenAI, Anthropic,
Google, and DeepSeek. The factory class uses a mapping of provider names to their respective classes and
provides methods to create and return the appropriate provider instance.

Classes:
    ProviderFactory: Factory class for creating LLM providers.

Usage example:
    factory = ProviderFactory()
    provider = factory.create("openai:gpt-4o-2024-08-06")
    response = provider.query(system_message="Hello", user_message="How are you?")
"""

from typing import Dict, Callable

from smolmodels.internal.common.providers.anthropic import AnthropicProvider
from smolmodels.internal.common.providers.deepseek import DeepSeekProvider
from smolmodels.internal.common.providers.google import GoogleProvider
from smolmodels.internal.common.providers.openai import OpenAIProvider
from smolmodels.internal.common.providers.provider import Provider


class ProviderFactory:
    """
    Factory class for LLM providers.
    """

    providers_map: Dict[str | None, Callable[[str | None], Provider]] = {
        "openai": lambda model: OpenAIProvider(model=model),
        "anthropic": lambda model: AnthropicProvider(model=model),
        "google": lambda model: GoogleProvider(model=model),
        "deepseek": lambda model: DeepSeekProvider(model=model),
        None: lambda model: OpenAIProvider(),
    }

    @staticmethod
    def create(provider_name: str | None = None) -> Provider:
        """
        Creates a provider based on the provider name. The provider name is expected to be in
        the format 'provider:model', where 'provider' is the name of the provider and 'model' is
        the name of the model. Valid input formats are 'provider', 'provider:model', or None.

        :param [str] provider_name: The name of the provider and model to use, in format 'provider:model'.
        :return [Provider]: The provider.
        """
        # Select default provider and model, if not specified
        if not provider_name:
            return ProviderFactory.providers_map[None](None)
        # Set selected provider with default model, if only provider is specified
        elif ":" not in provider_name:
            try:
                return ProviderFactory.providers_map[provider_name](None)
            except KeyError as e:
                raise ValueError(
                    f"Provider '{provider_name}' not supported, use one of {ProviderFactory.providers_map.keys()}."
                ) from e
        # Set selected provider and model, if both are specified
        elif (
            not provider_name.startswith(":")
            and not provider_name.endswith(":")
            and len(provider_and_model := provider_name.split(":")) == 2
        ):
            provider, model = provider_and_model
            try:
                return ProviderFactory.providers_map[provider](model)
            except KeyError as e:
                raise ValueError(
                    f"Provider '{provider}' not supported, use one of {ProviderFactory.providers_map.keys()}."
                ) from e
        # Raise error if provider and model are not specified correctly
        else:
            raise ValueError(
                f"Provider '{provider_name}' not supported, specify provider and model in the format 'provider:model' "
                f"with one of the supported providers: {ProviderFactory.providers_map.keys()}."
            )
