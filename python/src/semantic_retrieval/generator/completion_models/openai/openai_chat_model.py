from dataclasses import dataclass
from openai.api_resources.abstract.api_resource import APIResource
from typing import Optional, Union, List
from openai import api_resources

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


@dataclass
class OpenAIChatModelParams:
    completion_params: Optional[Union[CompletionModelParams, ChatCompletionCreateParams]] = None


@dataclass
class OpenAIChatModelConfig:
    api_key: Optional[str] = None
    default_model: Optional[str] = None


class OpenAIChatModel(CompletionModel):
    object_path = "/v1/chat/completions"
    resource_type = "chat.completion"

    def __init__(self, config: Optional[OpenAIChatModelConfig] = None):
        # TODO
        pass

    async def construct_messages(
        self, params: OpenAIChatModelParams
    ) -> List[ChatCompletionMessageParam]:
        messages = (
            params.completion_params.messages if params.completion_params is not None else []
        )

        content = str(params.prompt)

        messages.append({"role": "user", "content": content})
        return messages

    async def run(self, params: OpenAIChatModelParams) -> api_resources.ChatCompletion:
        completion_params = params.completion_params
        model = params.model if params.model is not None else self.default_model

        refined_completion_params = {
            "model": model,
            "messages": await self.construct_messages(params),
        }

        if getattr(completion_params, "stream", None) is not None:
            raise NotImplementedError("Streamed completions not implemented yet")
        else:
            response = self.client.api_requestor.request(
                "post",
                self.object_path,
                params=refined_completion_params,
                headers={"authorization": f"Bearer {self.api_key}"},
            )
            return api_resources.ChatCompletion.construct_from(response.data, self.client)
