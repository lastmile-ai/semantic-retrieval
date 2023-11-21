from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from semantic_retrieval.common.base import Attributable
from semantic_retrieval.common.json_types import JSONObject
from semantic_retrieval.common.types import CallbackEvent
from semantic_retrieval.document.document import Document, DocumentFragment
from semantic_retrieval.transformation.transformer import Transformer
from semantic_retrieval.utils.callbacks import CallbackManager, Traceable
from semantic_retrieval.utils.interop import canonical_field


class VectorEmbedding(Attributable):
    vector: List[
        float
    ] = []  # The vector representation of the embedding text.
    text: str = ""  # The text embedded via the vector.
    extras: Dict[Any, Any] = {
        "dimensions": 0,
        "min": 0,
        "max": 0,
    }  # Number of dimensions in the vector, and min/max values for each dimension.
    metadata: Optional[Dict[Any, Any]] = {
        canonical_field("document_id"): "",
        canonical_field("fragment_id"): "",
        "retrievalScore": 0.0,
    }  # Metadata


class ModelHandle(ABC):
    # e.g. openai.Embedding
    creator: Any
    # Contains a :
    # create: Callable[[Any], Any] = lambda *args, **kwargs: "result"  # type: ignore

    @staticmethod
    def mock(embedding: Optional[List[float]] = None) -> "ModelHandle":
        embedding_ = embedding or [0] * 1536

        class MockModelHandle(ModelHandle):
            class _MockResult:
                def to_dict_recursive(self):
                    return {"data": [{"embedding": embedding_}]}

            class _MockHandleCreator:
                def create(self, *args, **kwargs):  # type: ignore
                    return MockModelHandle._MockResult()

            creator: Any = _MockHandleCreator()

        return MockModelHandle()


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


class DocumentEmbeddingsTransformer(EmbeddingsTransformer, Traceable):
    def __init__(self, dimensions: int, callback_manager: CallbackManager):
        super().__init__(dimensions)
        self.callback_manager = callback_manager

    async def embed_fragment(
        self, fragment: DocumentFragment, model_handle: Optional[ModelHandle]
    ) -> VectorEmbedding:
        text = await fragment.get_content()
        metadata = {
            **(fragment.metadata or {}),
            "document_id": fragment.document_id,
            "fragment_id": fragment.fragment_id,
        }

        return await self.embed(
            text, model_handle=model_handle, metadata=metadata
        )

    async def embed_document(
        self, document: Document, model_handle: Optional[ModelHandle]
    ) -> List[VectorEmbedding]:
        embeddings = []
        for fragment in document.fragments:
            embeddings.append(
                await self.embed_fragment(fragment, model_handle)
            )

        await self.callback_manager.run_callbacks(
            CallbackEvent(
                name="document_embeddings_document_transformed",
                data=dict(
                    document=document,
                    embeddings=embeddings,
                ),
            )
        )
        return embeddings

    @abstractmethod
    async def transform_documents(
        self,
        documents: List[Document],
        model_handle: Optional[ModelHandle] = None,
    ) -> List[VectorEmbedding]:
        embeddings = []
        for document in documents:
            embeddings.extend(
                await self.embed_document(document, model_handle)
            )
        return embeddings
