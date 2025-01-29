"""
Unit tests for the ProviderFactory class.

This module contains unit tests for the ProviderFactory class, which is responsible for creating instances of different
LLM providers based on the specified provider name and model. The tests cover various scenarios including valid and invalid
provider names, model names, and edge cases.
"""

from unittest.mock import patch, MagicMock

import pytest

from smolmodels.internal.common.providers.provider import Provider
from smolmodels.internal.common.providers.provider_factory import ProviderFactory


def test_create_provider_with_model():
    with patch.dict(ProviderFactory.providers_map, {"openai": MagicMock()}) as mock_providers:
        provider_mock = mock_providers["openai"]
        provider_mock.return_value = MagicMock(spec=Provider)
        provider = ProviderFactory.create("openai:gpt-4o-2024-08-06")
        provider_mock.assert_called_once_with("gpt-4o-2024-08-06")
        assert isinstance(provider, Provider)


def test_create_provider_without_model():
    with patch.dict(ProviderFactory.providers_map, {"openai": MagicMock()}) as mock_providers:
        provider_mock = mock_providers["openai"]
        provider_mock.return_value = MagicMock(spec=Provider)
        provider = ProviderFactory.create("openai")
        provider_mock.assert_called_once_with(None)
        assert isinstance(provider, Provider)


def test_create_provider_with_invalid_provider():
    with pytest.raises(ValueError, match="Provider 'invalid_provider' not supported"):
        ProviderFactory.create("invalid_provider:model")


def test_create_provider_with_invalid_format():
    with pytest.raises(ValueError, match="Provider 'invalid:format:extra' not supported"):
        ProviderFactory.create("invalid:format:extra")


def test_create_provider_with_none():
    with patch.dict(ProviderFactory.providers_map, {None: MagicMock()}) as mock_providers:
        provider_mock = mock_providers[None]
        provider_mock.return_value = MagicMock(spec=Provider)
        provider = ProviderFactory.create(None)
        assert isinstance(provider, Provider)


def test_create_provider_with_empty_string():
    with patch.dict(ProviderFactory.providers_map, {None: MagicMock()}) as mock_providers:
        provider_mock = mock_providers[None]
        provider_mock.return_value = MagicMock(spec=Provider)
        provider = ProviderFactory.create("")
        assert isinstance(provider, Provider)


def test_create_provider_with_whitespace():
    with pytest.raises(ValueError, match="Provider ' openai' not supported"):
        ProviderFactory.create(" openai:gpt-4o ")


def test_create_provider_with_case_sensitivity():
    with pytest.raises(ValueError, match="Provider 'OpenAI' not supported"):
        ProviderFactory.create("OpenAI:gpt-4o")


def test_create_provider_with_special_characters():
    with patch.dict(ProviderFactory.providers_map, {"openai": MagicMock()}) as mock_providers:
        provider_mock = mock_providers["openai"]
        provider_mock.return_value = MagicMock(spec=Provider)
        provider = ProviderFactory.create("openai:gpt-4o@2024")
        provider_mock.assert_called_once_with("gpt-4o@2024")
        assert isinstance(provider, Provider)


def test_create_provider_with_colon_edge_cases():
    with pytest.raises(ValueError, match="Provider ':gpt-4o' not supported"):
        ProviderFactory.create(":gpt-4o")
    with pytest.raises(ValueError, match="Provider 'openai:' not supported"):
        ProviderFactory.create("openai:")


def test_create_provider_with_multiple_colons():
    with pytest.raises(ValueError, match="Provider 'openai:enterprise:gpt-4o' not supported"):
        ProviderFactory.create("openai:enterprise:gpt-4o")


def test_create_provider_with_invalid_data_types():
    with pytest.raises(TypeError):
        ProviderFactory.create(42)
    with pytest.raises(TypeError):
        ProviderFactory.create(["openai", "gpt-4o"])
    with pytest.raises(TypeError):
        ProviderFactory.create({"provider": "openai"})
