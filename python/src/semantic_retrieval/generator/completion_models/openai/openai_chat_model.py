from dataclasses import dataclass
from typing import List, Optional

from semantic_retrieval.generator.completion_models.completion_model import (
    CompletionModel,
    CompletionModelParams,
)


@dataclass
class ChatCompletionCreateParams:
    pass


@dataclass
class ChatCompletionMessageParam:
    pass


OpenAIChatModelParams = CompletionModelParams[ChatCompletionCreateParams]


@dataclass
class OpenAIChatModelConfig:
    api_key: Optional[str] = None
    default_model: Optional[str] = None


class OpenAIChatModel(CompletionModel[ChatCompletionCreateParams, ChatCompletion]):  # type: ignore
    object_path = "/v1/chat/completions"
    resource_type = "chat.completion"

    def __init__(self, config: Optional[OpenAIChatModelConfig] = None):
        pass

    async def construct_messages(
        self, params: OpenAIChatModelParams
    ) -> List[ChatCompletionMessageParam]:
        raise NotImplementedError()

    async def run(self, params: OpenAIChatModelParams) -> ChatCompletion:  # type: ignore
        raise NotImplementedError()
