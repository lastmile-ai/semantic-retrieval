from typing import Dict, Any, List
from semantic_retrieval.common.types import Record

from semantic_retrieval.transformation.embeddings.embeddings import (
    VectorEmbedding,
    EmbeddingsTransformer,
)

from semantic_retrieval.document.metadata.document_metadata_db import DocumentMetadataDB

from semantic_retrieval.utils.callbacks import Traceable


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
    ):
        self.embeddings = embeddings
        self.metadata_db = metadata_db
        self.callback_manager = None
        self.vector_db_config = vector_db_config

    @classmethod
    def fromDocuments(cls, documents, config) -> "VectorDB":  # type: ignore [fixme]
        # TODO implement
        pass

    async def addDocuments(self, documents) -> None:  # type: ignore [fixme]
        # TODO implement
        pass

    async def query(self, query: VectorDBQuery) -> List[VectorEmbedding]:
        # TODO implement
        return []
