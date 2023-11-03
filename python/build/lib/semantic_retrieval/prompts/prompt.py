from abc import ABC, abstractmethod


class IPrompt(ABC):
    @abstractmethod
    async def to_string(self) -> str:
        pass


class Prompt(IPrompt):
    def __init__(self, prompt: str):
        self.prompt = prompt

    async def to_string(self) -> str:
        return self.prompt
