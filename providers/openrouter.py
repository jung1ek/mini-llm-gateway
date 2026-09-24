from openai import AsyncOpenAI
from langchain_openai import ChatOpenAI

from mini_llm_gateway.providers.base import BaseLLM


class OpenRouterLLM(BaseLLM):

    def __init__(self, **kwargs):
        self.client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", **kwargs)

    async def generate(self, messages, **kwargs):
        return await self.client.chat.completions.create(
            model="openrouter/free", messages=messages, **kwargs
        )


class OpenRouterLLM(BaseLLM):

    def __init__(self, **kwargs):
        self.client = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1", model="openrouter/free", **kwargs
        )

    async def generate(self, messages, **kwargs):
        return await self.client.ainvoke(messages, **kwargs)
