from mini_llm_gateway.providers.base import BaseLLM


class LLMGateway:

    def __init__(self, providers: list[BaseLLM], **kwargs):
        self.providers = providers

    async def generate(self, messages: list[dict[str, str]], **kwargs):
        last_error = None

        for llm in self.providers:
            try:
                print(f"Using {llm.__class__.__name__}")
                response = await llm.generate(messages, **kwargs)
                return response

            except Exception as e:
                print(f"Failed {e}")
                last_error = e

        raise RuntimeError("All Providers failed") from last_error
