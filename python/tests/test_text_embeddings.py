import pytest

from semantic_retrieval.transformation.embeddings.openai_embeddings import (
    OpenAIEmbeddings,
    OpenAIEmbeddingsConfig,
)


@pytest.mark.asyncio
async def test_openai_emb_query():
    cfg = OpenAIEmbeddingsConfig(
        # TODO: do this properly
        api_key_path_abs="/Users/jonathan/keys/dev_OPENAI_API_KEY.txt"
    )
    e = OpenAIEmbeddings(cfg)
    res = await e.embed("hello world")
    dim = len(res.vector)
    assert dim == 1536
