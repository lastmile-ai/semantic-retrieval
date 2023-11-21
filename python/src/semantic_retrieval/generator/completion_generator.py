from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, Optional

from semantic_retrieval.common.types import P, R
from semantic_retrieval.generator.completion_models.completion_model import (
    CompletionModel,
    CompletionModelParams,
)
from semantic_retrieval.utils.callbacks import CallbackManager, Traceable


@dataclass
class LLMCompletionGeneratorParams(Generic[P], CompletionModelParams[P]):
    pass


class LLMCompletionGenerator(Generic[P, R], Traceable):
    def __init__(
        self,
        model: CompletionModel[P, R],
        callback_manager: Optional[CallbackManager] = None,
    ):
        self.model = model
        self.callback_manager = callback_manager

    @abstractmethod
    async def run(self, params: P) -> R:
        pass
