from typing import List

import pinecone
from pinecone import ScoredVector
from semantic_retrieval.common.types import Record
from semantic_retrieval.data_store.vector_dbs.vector_db import (
    VectorDB,
    VectorDBConfig,
    VectorDBEmbeddingQuery,
    VectorDBQuery,
    VectorDBTextQuery,
)
from semantic_retrieval.document.document import Document
from semantic_retrieval.document.metadata.document_metadata_db import DocumentMetadataDB
from semantic_retrieval.transformation.embeddings.embeddings import (
    DocumentEmbeddingsTransformer,
    VectorEmbedding,
)


class PineconeVectorDBConfig(VectorDBConfig):
    index_name: str
    api_key: str
    environment: str
    namespace: str


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
        # TODO impl
        pass

    async def query(self, query: VectorDBQuery) -> List[VectorEmbedding]:
        async def _get_query_vector():
            match query:
                case VectorDBEmbeddingQuery(
                    embedding_vector=vec,
                ):
                    return vec
                case VectorDBTextQuery(text=text):
                    return await self.embeddings.embed(
                        text=text, model_handle=None, metadata=None
                    )

        vec = await _get_query_vector()

        pinecone.init(api_key=self.config.api_key, environment=self.config.environment)
        index = pinecone.Index(self.config.index_name)

        top_k = query.topK
        metadata_filter = query.metadata_filter

        query_response = index.query(
            namespace=self.config.namespace,
            top_k=top_k,
            include_values=True,
            include_metadata=True,
            vector=vec.vector,
            filter=metadata_filter,
        )

        # TODO type better
        def _response_record_to_vector_embedding(
            match: ScoredVector,
        ) -> VectorEmbedding:
            return VectorEmbedding(
                vector=match.values[:10],
                metadata=match.metadata,
                text=match.metadata["text"],
            )

        return list(map(_response_record_to_vector_embedding, query_response.matches))
