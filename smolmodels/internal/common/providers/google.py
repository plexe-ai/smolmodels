import logging
import google.generativeai as genai
import os

from smolmodels.internal.common.providers.provider import Provider


logger = logging.getLogger(__name__)


class GoogleProvider(Provider):
    def __init__(self, api_key: str = None, model: str = "gemini-1.5-flash"):
        self.key = api_key or os.environ.get("GOOGLE_API_KEY", default=None)
        self.model = model
        self.generation_config = genai.GenerationConfig(max_output_tokens=4096)

    def query(self, system_message: str, user_message: str, response_format=None) -> str:
        logger.debug(f"Requesting chat completion from {self.model} with messages: {system_message}, {user_message}")

        llm = genai.GenerativeModel(
            model_name=self.model,
            generation_config=self.generation_config,
            system_instruction=system_message,
        )

        if response_format is not None:
            raise NotImplementedError("Google GenAI does not support function calling for now.")
        else:
            response = llm.generate_content(user_message)
        logger.debug(f"Received completion from {self.model}: {response.text}")
        return response.text
