from typing import Any


from semantic_retrieval.utils.callbacks import (
    CallbackManager,
    Traceable,
)


class CompletionModelParams:
    def __init__(self, prompt: str or IPrompt, model: str = None, completionParams: Any = None):
        self.prompt = prompt
        self.model = model
        self.completionParams = completionParams


class CompletionModel(Traceable):
    def __init__(self, callback_manager: CallbackManager = None):
        self.callback_manager = callback_manager

    async def run(self, params: CompletionModelParams):
        # Implement this method to interact with different LLM completion models
        pass
