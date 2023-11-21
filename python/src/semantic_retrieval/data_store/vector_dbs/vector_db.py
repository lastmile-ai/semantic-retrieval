from abc import abstractmethod
from typing import Any, Dict, List, Sequence

from semantic_retrieval.common.types import Record
from semantic_retrieval.document.document import Document
from semantic_retrieval.document.metadata.document_metadata_db import (
    DocumentMetadataDB,
)
from semantic_retrieval.transformation.embeddings.embeddings import (
    EmbeddingsTransformer,
    VectorEmbedding,
)
from semantic_retrieval.utils.callbacks import CallbackManager, Traceable


class VectorDBBaseQuery(Record):
    mode: str
    metadata_filter: Dict[str, Any]
    topK: int


class VectorDBEmbeddingQuery(VectorDBBaseQuery):
    embedding_vector: VectorEmbedding


class VectorDBTextQuery(VectorDBBaseQuery):
    text: str


VectorDBQuery = VectorDBEmbeddingQuery | VectorDBTextQuery


class VectorDBConfig(Record):
    pass


def is_embedding_query(query: VectorDBQuery):
    return hasattr(query, "embedding_vector")


def isTextQuery(query: VectorDBQuery):
    return hasattr(query, "text")


class VectorDB(Traceable):
    def __init__(
        self,
        embeddings: EmbeddingsTransformer,
        metadata_db: DocumentMetadataDB,
        vector_db_config: VectorDBConfig,
        callback_manager: CallbackManager,
    ):
        self.embeddings = embeddings
        self.metadata_db = metadata_db
        self.callback_manager = callback_manager
        self.vector_db_config = vector_db_config

    @classmethod
    @abstractmethod
    def fromDocuments(
        cls, documents: Sequence[Document], config: VectorDBConfig
    ) -> "VectorDB":
        pass

    @abstractmethod
    async def addDocuments(self, documents: Sequence[Document]) -> None:
        pass

    @abstractmethod
    async def query(
        self,
        query: VectorDBQuery,
    ) -> List[VectorEmbedding]:
        pass
