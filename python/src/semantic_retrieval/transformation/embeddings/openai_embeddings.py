from dataclasses import dataclass
from typing import List, Optional, Any, Dict
from semantic_retrieval.common.json_types import JSONObject

# from openai import OpenAI, EmbeddingsResponse

from semantic_retrieval.transformation.embeddings.embeddings import (
    DocumentEmbeddingsTransformer,
    VectorEmbedding,
)

from semantic_retrieval.document.document import Document


class OpenAIClientOptions:
    # TODO: import openai
    pass


class OpenAI:
    # TODO: import
    def __init__(self, **kwargs) -> None:
        pass


def getEnvVar(key):
    # TODO
    return "THE_VAR_VALUE"


@dataclass
class OpenAIEmbeddingsConfig(OpenAIClientOptions):
    apiKey: Optional[str]


class OpenAIEmbeddings(DocumentEmbeddingsTransformer):
    DEFAULT_MODEL = "text-embedding-ada-002"
    MODEL_DIMENSIONS = {
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, config: Optional[OpenAIEmbeddingsConfig] = None):
        super().__init__(self.MODEL_DIMENSIONS[self.DEFAULT_MODEL])
        # TOD
        pass

    def embed(self, text: str, metadata: Optional[JSONObject] = None) -> VectorEmbedding:
        # TODO
        pass

    async def transform_documents(self, documents: List[Document]) -> List[VectorEmbedding]:
        # TODO
        pass

    async def create_embeddings(self, fragments: List[Dict]) -> List[VectorEmbedding]:
        # TODO
        pass
