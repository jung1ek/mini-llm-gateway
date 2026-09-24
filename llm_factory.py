from typing import Literal

from mini_llm_gateway.providers.base import BaseLLM
from mini_llm_gateway.providers.gemini import GeminiLLM
from mini_llm_gateway.providers.groq import GroqLLM
from mini_llm_gateway.providers.openrouter import OpenRouterLLM


def get_llm(provider: Literal["gemini", "groq", "openrouter"], model: str) -> BaseLLM:

    match provider:
        case "gemini":
            return GeminiLLM()
        case "groq":
            return GroqLLM()
        case "openrouter":
            return OpenRouterLLM()
        case _:
            pass
