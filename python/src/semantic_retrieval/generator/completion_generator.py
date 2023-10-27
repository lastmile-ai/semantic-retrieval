from semantic_retrieval.utils.callbacks import (
    CallbackManager,
    Traceable,
)

from semantic_retrieval.generator.completion_models.completion_model import (
    CompletionModel,
    CompletionModelParams,
)


class LLMCompletionGeneratorParams(CompletionModelParams):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class LLMCompletionGenerator(Traceable):
    def __init__(self, model: CompletionModel, callback_manager: CallbackManager = None):
        self.model = model
        self.callback_manager = callback_manager

    async def run(self, params: LLMCompletionGeneratorParams):
        # Implement this method to perform completion generation using the given parameters
        pass
