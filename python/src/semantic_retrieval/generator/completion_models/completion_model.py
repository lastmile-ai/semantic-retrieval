from dataclasses import dataclass
from typing import Any, Awaitable, Optional
from semantic_retrieval.common.types import R
from semantic_retrieval.prompts.prompt import IPrompt


from semantic_retrieval.utils.callbacks import (
    CallbackManager,
    Traceable,
)


@dataclass
class CompletionModelParams:
    pass


class CompletionModel(Traceable):
    def __init__(self, callback_manager: Optional[CallbackManager] = None):
        self.callback_manager = callback_manager

    async def run(self, params: CompletionModelParams) -> Awaitable[R]:
        # Implement this method to interact with different LLM completion models
        pass
