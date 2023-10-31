from abc import abstractmethod
from typing import Any, Dict, List, Optional

from semantic_retrieval.common.base import Attributable

from semantic_retrieval.transformation.transformer import Transformer

from semantic_retrieval.common.json_types import JSONObject

from semantic_retrieval.document.document import Document, DocumentFragment


class VectorEmbedding(Attributable):
    vector: List[float] = []  # The vector representation of the embedding text.
    text: str = ""  # The text embedded via the vector.
    extras: Dict[Any, Any] = {
        "dimensions": 0,
        "min": 0,
        "max": 0,
    }  # Number of dimensions in the vector, and min/max values for each dimension.
    metadata: Optional[Dict[Any, Any]] = {
        "document_id": "",
        "fragmentId": "",
        "retrievalScore": 0.0,
    }  # Metadata


class ModelHandle(Attributable):
    # like openai.Embedding
    embedding_creator: Any  # TODO this better


class EmbeddingsTransformer(Transformer):
    def __init__(self, dimensions: int):
        self.dimensions = dimensions

    @abstractmethod
    async def embed(self, model_handle: ModelHandle, text: str, metadata: JSONObject) -> VectorEmbedding:  # type: ignore [fixme]
        pass


class DocumentEmbeddingsTransformer(EmbeddingsTransformer):
    def __init__(self, dimensions: int):
        super().__init__(dimensions)

    async def embedFragment(
        self, model_handle: ModelHandle, fragment: DocumentFragment
    ) -> VectorEmbedding:
        text = await fragment.get_content()
        metadata = {
            **(fragment.metadata or {}),
            "document_id": fragment.document_id,
            "fragmentId": fragment.fragment_id,
        }

        return await self.embed(model_handle, text, metadata)

    async def embedDocument(
        self, model_handle: ModelHandle, document: Document
    ) -> List[VectorEmbedding]:
        embeddings = []
        for fragment in document.fragments:
            embeddings.append(await self.embedFragment(model_handle, fragment))
        return embeddings

    async def transformDocuments(
        self, model_handle: ModelHandle, documents: List[Document]
    ) -> List[VectorEmbedding]:
        embeddings = []
        for document in documents:
            embeddings.extend(await self.embedDocument(model_handle, document))
        return embeddings
