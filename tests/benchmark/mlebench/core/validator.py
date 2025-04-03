"""Environment validation for MLE Bench runner."""

import os
import sys


class EnvironmentValidator:
    """Class to validate environment setup"""

    @staticmethod
    def ensure_kaggle_credentials():
        """Ensure that Kaggle API credentials are set up"""
        print("🔑 Checking Kaggle API credentials...")
        if not os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
            print(
                "❌ Kaggle API credentials not found. Please save 'kaggle.json' to '~/.kaggle/' following "
                "the instructions at https://www.kaggle.com/docs/api."
            )
            sys.exit(1)
        print("✅ Kaggle API credentials found.")

    @staticmethod
    def check_llm_api_keys():
        """Check if required LLM API key environment variables are set"""
        # Check for common LLM provider API keys
        api_keys = {"OpenAI": "OPENAI_API_KEY", "Anthropic": "ANTHROPIC_API_KEY", "Gemini": "GEMINI_API_KEY"}

        keys_found = False
        for provider, env_var in api_keys.items():
            if os.environ.get(env_var):
                print(f"✅ {provider} API key found ({env_var})")
                keys_found = True

        if not keys_found:
            print("❌ No LLM API keys found. Please set one of the following environment variables:")
            for provider, env_var in api_keys.items():
                print(f"❌   - {env_var} (for {provider})")
            return False

        return True
