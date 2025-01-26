import logging
import anthropic
import os

from smolmodels.internal.common.providers.provider import Provider


logger = logging.getLogger(__name__)


class AnthropicProvider(Provider):
    def __init__(self, api_key: str = None, model: str = "claude-3-5-sonnet-20241022"):
        self.key = api_key or os.environ.get("ANTHROPIC_API_KEY", default=None)
        self.model = model
        self.max_tokens = 4096
        self.client = anthropic.Anthropic(api_key=self.key)

    def query(self, system_message: str, user_message: str, response_format=None) -> str:
        logger.debug(f"Requesting chat completion from {self.model} with messages: {system_message}, {user_message}")
        if response_format is not None:
            raise NotImplementedError("Anthropic does not support response format parsing.")
        # todo: implement https://python.useinstructor.com/blog/2024/10/23/structured-outputs-and-prompt-caching-with-anthropic/#structured-outputs-with-anthropic-and-instructor
        else:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
            )
            content = response.content
        logger.debug(f"Received completion from {self.model}: {content}")
        return content
