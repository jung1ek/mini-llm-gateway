from google import genai
from google.genai import types
from google.genai.types import AutomaticFunctionCallingConfig
from langchain_core.messages import AIMessage
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

from mini_llm_gateway.providers.base import BaseLLM

# class GeminiLLM(BaseLLM):

#     def __init__(self):
#         self.client = genai.Client()

#     async def generate(self, messages):
#         history = [
#             types.Content(
#                 role="model" if message["role"] == "assistant" else message["role"],
#                 parts=[types.Part(text=message["content"])],
#             )
#             for message in messages[:-1]
#         ]

#         chat = self.client.aio.chats.create(
#             model="gemini-3.1-flash-lite",
#             history=history,
#         )

#         response = await chat.send_message(message=messages[-1]["content"])

#         return response


def normalize_response(response: AIMessage) -> AIMessage:
    if isinstance(response.content, list):
        text_parts = []

        for block in response.content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))

        return AIMessage(
            content="".join(text_parts),
            additional_kwargs=response.additional_kwargs,
            response_metadata=response.response_metadata,
            id=response.id,
        )

    return response


class GeminiLLM(BaseLLM):

    def __init__(self, **kwargs):
        self.client = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite", **kwargs
        ).bind(automatic_function_calling=AutomaticFunctionCallingConfig(disable=True))

    async def generate(self, messages, **kwargs):
        response = await self.client.ainvoke(messages, **kwargs)
        return normalize_response(response)
