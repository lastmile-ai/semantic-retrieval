import pytest

from semantic_retrieval.transformation.embeddings.openai_embeddings import (
    OpenAIEmbeddings,
    OpenAIEmbeddingsConfig,
)

from dotenv import load_dotenv
import os

load_dotenv()


@pytest.mark.asyncio
async def test_openai_emb_query():
    cfg = OpenAIEmbeddingsConfig(api_key=os.getenv("OPENAI_API_KEY"))
    e = OpenAIEmbeddings(cfg)
    res = await e.embed("hello world")
    dim = len(res.vector)
    assert dim == 1536
