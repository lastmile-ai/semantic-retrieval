from typing import Any
import pytest
from semantic_retrieval.transformation.embeddings.embeddings import ModelHandle

from semantic_retrieval.transformation.embeddings.openai_embeddings import (
    OpenAIEmbeddings,
    OpenAIEmbeddingsConfig,
)
from semantic_retrieval.utils.callbacks import CallbackManager


class _MockResult:
    def to_dict_recursive(self):
        return {"data": [{"embedding": [0] * 1536}]}


class _MockHandleCreator:
    def create(self, *args, **kwargs):  # type: ignore
        return _MockResult()


class MockModelHandle(ModelHandle):
    creator: Any = _MockHandleCreator()


@pytest.mark.asyncio
async def test_openai_emb_query():
    cfg = OpenAIEmbeddingsConfig(api_key="mock_key")
    e = OpenAIEmbeddings(cfg, callback_manager=CallbackManager.default())

    model_handle = MockModelHandle()
    res = await e.embed("hello world", model_handle=model_handle)
    dim = len(res.vector)
    assert dim == 1536
