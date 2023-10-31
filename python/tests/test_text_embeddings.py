from typing import Any, Callable
import pytest
from semantic_retrieval.transformation.embeddings.embeddings import ModelHandle

from semantic_retrieval.transformation.embeddings.openai_embeddings import (
    OpenAIEmbeddings,
    OpenAIEmbeddingsConfig,
)

from dotenv import load_dotenv
import os


class _MockResult:
    def to_dict_recursive(self):
        return {"data": [{"embedding": [0] * 1536}]}


class MockModelHandle(ModelHandle):
    create: Callable[[Any], Any] = lambda *args, **kwargs: _MockResult()  # type: ignore


@pytest.mark.asyncio
async def test_openai_emb_query():
    load_dotenv()
    cfg = OpenAIEmbeddingsConfig(api_key=os.getenv("OPENAI_API_KEY"))
    e = OpenAIEmbeddings(cfg)

    model_handle = MockModelHandle()
    res = await e.embed(model_handle, "hello world")
    dim = len(res.vector)
    assert dim == 1536
