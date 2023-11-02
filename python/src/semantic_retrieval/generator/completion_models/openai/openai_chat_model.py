from dataclasses import dataclass
from typing import Optional, List
from openai import ChatCompletion

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


class OpenAIChatModel(CompletionModel[ChatCompletionCreateParams, ChatCompletion]):
    object_path = "/v1/chat/completions"
    resource_type = "chat.completion"

    def __init__(self, config: Optional[OpenAIChatModelConfig] = None):
        # TODO [P0] impl
        pass

    async def construct_messages(
        self, params: OpenAIChatModelParams
    ) -> List[ChatCompletionMessageParam]:
        # TODO [P0] imple w/ aiconfig?
        messages = []
        return messages

    async def run(self, params: OpenAIChatModelParams) -> ChatCompletion:  # type: ignore [fixme]
        # TODO [P0] impl w/ aiconfig?
        pass
