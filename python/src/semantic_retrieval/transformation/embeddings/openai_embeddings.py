from dataclasses import dataclass
from typing import List, Optional
from semantic_retrieval.common.json_types import JSONObject
from semantic_retrieval.common.types import Record

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
    def __init__(self, **kwargs) -> None:  # type: ignore [fixme]
        pass


class EmbedFragmentData(Record):
    document_id: str
    fragment_id: str
    text: str


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

    def embed(self, text: str, metadata: Optional[JSONObject] = None) -> VectorEmbedding:  # type: ignore [fixme]
        # TODO
        pass

    async def transform_documents(self, documents: List[Document]) -> List[VectorEmbedding]:  # type: ignore [fixme]
        # TODO
        pass

    async def create_embeddings(self, fragments: List[EmbedFragmentData]) -> List[VectorEmbedding]:  # type: ignore [fixme]
        # TODO
        pass
