import pytest
from semantic_retrieval.transformation.embeddings.embeddings import ModelHandle

from semantic_retrieval.transformation.embeddings.openai_embeddings import (
    OpenAIEmbeddings,
    OpenAIEmbeddingsConfig,
)
from semantic_retrieval.utils.callbacks import CallbackManager


@pytest.mark.asyncio
async def test_openai_emb_query():
    cfg = OpenAIEmbeddingsConfig(api_key="mock_key")
    e = OpenAIEmbeddings(cfg, callback_manager=CallbackManager.default())

    model_handle = ModelHandle.mock()
    res = await e.embed("hello world", model_handle=model_handle)
    dim = len(res.vector)
    assert dim == 1536
