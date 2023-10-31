from typing import List, Optional
from semantic_retrieval.common.types import Record

from semantic_retrieval.data_store.vector_dbs.vector_db import (
    VectorDB,
    VectorDBConfig,
    VectorDBQuery,
)

from semantic_retrieval.document.document import Document
from semantic_retrieval.document.metadata.document_metadata_db import DocumentMetadataDB
from semantic_retrieval.transformation.embeddings.embeddings import (
    DocumentEmbeddingsTransformer,
    VectorEmbedding,
)


class PineconeVectorDBConfig(VectorDBConfig):
    index_name: str
    api_key: Optional[str] = None
    environment: Optional[str] = None
    namespace: Optional[str] = None


class PineconeVectorDB(VectorDB):
    def __init__(
        self,
        config: PineconeVectorDBConfig,
        embeddings: DocumentEmbeddingsTransformer,
        metadata_db: DocumentMetadataDB,
    ):
        self.config = config
        self.embeddings = embeddings
        self.metadata_db = metadata_db

    @classmethod
    async def from_documents(
        cls,
        documents: List[Document],
        config: PineconeVectorDBConfig,
        embeddings: DocumentEmbeddingsTransformer,
        metadata_db: DocumentMetadataDB,
    ):
        instance = cls(config, embeddings, metadata_db)
        await instance.add_documents(documents)
        return instance

    def sanitize_metadata(self, unsanitized_metadata: Record):
        # TODO
        pass

    async def add_documents(self, documents: List[Document]):
        
        # Suyog to write this & also need to update fn signature


        pass

    async def query(self, query: VectorDBQuery) -> List[VectorEmbedding]:  # type: ignore [fixme]
        # TODO impl
        return []
