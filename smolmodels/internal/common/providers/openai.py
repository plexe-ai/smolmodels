import logging
import openai
import os

from smolmodels.internal.common.providers.provider import Provider


logger = logging.getLogger(__name__)


class OpenAIProvider(Provider):
    def __init__(self, api_key: str = None, model: str = "gpt-4o-2024-08-06"):
        self.key = api_key or os.environ.get("OPENAI_API_KEY", default=None)
        self.model = model
        self.client = openai.OpenAI(api_key=self.key)

    def query(self, system_message: str, user_message: str, response_format=None) -> str:
        logger.debug(f"Requesting chat completion from {self.model} with messages: {system_message}, {user_message}")
        if response_format is not None:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                response_format=response_format,
            )
            content = response.choices[0].message.content
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
            )
            content = response.choices[0].message.content
        logger.debug(f"Received completion from {self.model}: {content}")
        return content
