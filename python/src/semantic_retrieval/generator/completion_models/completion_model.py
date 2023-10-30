from typing import Generic, Optional

from semantic_retrieval.common.types import P, R, Record
from semantic_retrieval.prompts.prompt import IPrompt
from semantic_retrieval.utils.callbacks import CallbackManager


class CompletionModelParams(Generic[P], Record):
    def __init__(
        self,
        prompt: str | IPrompt,
        model: Optional[str] = None,
        completion_params: Optional[P] = None,
    ):
        self.prompt = prompt
        self.model = model
        self.completion_params = completion_params


class CompletionModel(Generic[P, R]):
    callback_manager = None

    def __init__(self, callback_manager: Optional[CallbackManager] = None):
        self.callback_manager = callback_manager

    async def run(self, params: CompletionModelParams[P]) -> R:
        raise NotImplementedError
