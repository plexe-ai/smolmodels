import logging
import openai
import os

from smolmodels.internal.common.providers.provider import Provider


logger = logging.getLogger(__name__)


class DeepSeekProvider(Provider):
    def __init__(self, api_key: str = None, model: str = "deepseek-chat"):
        self.key = api_key or os.environ.get("DEEPSEEK_API_KEY", default=None)
        self.model = model
        self.client = openai.OpenAI(api_key=self.key, base_url="https://api.deepseek.com")

    def query(self, system_message: str, user_message: str, response_format=None) -> str:
        self._log_request(system_message, user_message, self.model, logger)

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

        self._log_response(content, self.model, logger)

        return content
