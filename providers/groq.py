from groq import AsyncGroq
from langchain_groq.chat_models import ChatGroq

from mini_llm_gateway.providers.base import BaseLLM


class GroqLLM(BaseLLM):

    def __init__(self):
        self.client = AsyncGroq()

    async def generate(self, messages):
        response = await self.client.chat.completions.create(
            model="openai/gpt-oss-120b", messages=messages
        )
        return response


class GroqLLM(BaseLLM):

    def __init__(self, **kwargs):
        self.client = ChatGroq(model="openai/gpt-oss-120b", **kwargs)

    async def generate(self, messages, **kwargs):
        return await self.client.ainvoke(messages, **kwargs)
