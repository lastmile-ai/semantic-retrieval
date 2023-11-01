from abc import ABC, abstractmethod
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


class ModelHandle(ABC):
    # e.g. openai.Embedding
    creator: Any
    # Contains a :
    # create: Callable[[Any], Any] = lambda *args, **kwargs: "result"  # type: ignore


class EmbeddingsTransformer(Transformer):
    def __init__(self, dimensions: int):
        self.dimensions = dimensions

    @abstractmethod
    async def embed(
        self,
        text: str,
        model_handle: Optional[ModelHandle],
        metadata: Optional[JSONObject] = None,
    ) -> VectorEmbedding:
        pass


class DocumentEmbeddingsTransformer(EmbeddingsTransformer):
    def __init__(self, dimensions: int):
        super().__init__(dimensions)

    async def embed_fragment(
        self, fragment: DocumentFragment, model_handle: Optional[ModelHandle]
    ) -> VectorEmbedding:
        text = await fragment.get_content()
        metadata = {
            **(fragment.metadata or {}),
            "document_id": fragment.document_id,
            "fragmentId": fragment.fragment_id,
        }

        return await self.embed(text, model_handle=model_handle, metadata=metadata)

    async def embed_document(
        self, document: Document, model_handle: Optional[ModelHandle]
    ) -> List[VectorEmbedding]:
        embeddings = []
        for fragment in document.fragments:
            embeddings.append(await self.embed_fragment(fragment, model_handle))
        return embeddings

    @abstractmethod
    async def transform_documents(
        self,
        documents: List[Document],
        model_handle: Optional[ModelHandle] = None,
    ) -> List[VectorEmbedding]:
        embeddings = []
        for document in documents:
            embeddings.extend(await self.embed_document(document, model_handle))
        return embeddings
