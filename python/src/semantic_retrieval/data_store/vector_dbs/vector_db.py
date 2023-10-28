from dataclasses import dataclass
from typing import Dict, Any

from semantic_retrieval.transformation.embeddings.embeddings import (
    VectorEmbedding,
    EmbeddingsTransformer,
)

from semantic_retrieval.document.metadata.document_metadata_db import DocumentMetadataDB

from semantic_retrieval.utils.callbacks import Traceable, CallbackManager


class VectorDBBaseQuery:
    mode: str
    metadataFilter: Dict[str, Any]
    topK: int


class VectorDBEmbeddingQuery(VectorDBBaseQuery):
    embeddingVector: VectorEmbedding


class VectorDBTextQuery(VectorDBBaseQuery):
    text: str


@dataclass
class VectorDBConfig:
    embeddings: EmbeddingsTransformer
    metadata_db: DocumentMetadataDB


def isEmbeddingQuery(query):
    return hasattr(query, "embeddingVector")


def isTextQuery(query):
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
    def fromDocuments(cls, documents, config):
        raise Exception("VectorDB implementation missing override")

    def addDocuments(self, documents):
        raise Exception("VectorDB implementation missing override")

    def query(self, query):
        raise Exception("VectorDB implementation missing override")
