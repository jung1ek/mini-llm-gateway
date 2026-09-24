from abc import ABC, abstractmethod


class BaseLLM(ABC):

    @abstractmethod
    async def generate(self, messages):
        raise NotImplementedError
