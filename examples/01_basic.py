import asyncio
from dotenv import load_dotenv

from mini_llm_gateway.providers.groq import GroqLLM
from mini_llm_gateway.providers.gemini import GeminiLLM
from mini_llm_gateway.providers.openrouter import OpenRouterLLM
from mini_llm_gateway.gateway import LLMGateway

load_dotenv()


async def main():

    llm = LLMGateway(providers=[OpenRouterLLM(), GroqLLM(), GeminiLLM()])

    result = await llm.generate(
        [{"role": "user", "content": "Explain AI agents in 3 sentences."}]
    )

    print(result)


if __name__ == "__main__":
    asyncio.run(main())
