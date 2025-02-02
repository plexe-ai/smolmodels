"""
Unit tests for the ProviderFactory class.

This module contains unit tests for the ProviderFactory class, which is responsible for creating instances of
LiteLLM providers. The tests cover various scenarios including valid and invalid provider names, model names,
and edge cases.
"""

from unittest.mock import patch

import pytest

from smolmodels.internal.common.providers.provider_factory import ProviderFactory, LiteLLMProvider


def test_create_provider_with_model():
    provider = ProviderFactory.create("openai:gpt-4o-mini")
    assert isinstance(provider, LiteLLMProvider)
    assert provider.model == "openai/gpt-4o-mini"


def test_create_provider_without_model():
    provider = ProviderFactory.create("openai")
    assert isinstance(provider, LiteLLMProvider)
    assert provider.model == "openai"


def test_create_provider_with_none():
    provider = ProviderFactory.create(None)
    assert isinstance(provider, LiteLLMProvider)
    assert provider.model == "openai/gpt-4o-mini"


def test_create_provider_with_empty_string():
    provider = ProviderFactory.create("")
    assert isinstance(provider, LiteLLMProvider)
    assert provider.model == "openai/gpt-4o-mini"


def test_create_provider_with_whitespace():
    with pytest.raises(ValueError):
        ProviderFactory.create(" openai:gpt-4o ")


def test_create_provider_with_case_sensitivity():
    # Case sensitivity should be preserved for model names
    provider = ProviderFactory.create("OpenAI:GPT-4")
    assert isinstance(provider, LiteLLMProvider)
    assert provider.model == "OpenAI/GPT-4"


def test_create_provider_with_special_characters():
    provider = ProviderFactory.create("openai:gpt-4o@2024")
    assert isinstance(provider, LiteLLMProvider)
    assert provider.model == "openai/gpt-4o@2024"


def test_create_provider_with_colon_edge_cases():
    with pytest.raises(ValueError):
        ProviderFactory.create(":gpt-4o")
    with pytest.raises(ValueError):
        ProviderFactory.create("openai:")


def test_create_provider_with_multiple_colons():
    with pytest.raises(ValueError):
        ProviderFactory.create("openai:enterprise:gpt-4o")


def test_create_provider_with_invalid_data_types():
    with pytest.raises(TypeError):
        ProviderFactory.create(42)
    with pytest.raises(TypeError):
        ProviderFactory.create(["openai", "gpt-4o"])
    with pytest.raises(TypeError):
        ProviderFactory.create({"provider": "openai"})
