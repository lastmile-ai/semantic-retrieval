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
from semantic_retrieval.transformation.embeddings.openai_embeddings import (
    OpenAIEmbeddings,
    OpenAIEmbeddingsConfig,
)
from typing import Generator, Any, Tuple
import itertools


# This is directly from pinecone's documentation: https://docs.pinecone.io/docs/insert-data#batching-upserts
def chunks(
    iterable: List[Any], batch_size: int = 100
) -> Generator[List[Tuple[Any, Any]], Any, None]:
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk  # type: ignore
        chunk = tuple(itertools.islice(it, batch_size))


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

    async def add_documents(
        self,
        documents: List[Document],
    ):
        pinecone.init(api_key=self.config.api_key, environment=self.config.environment)
        index = pinecone.Index(self.config.index_name)

        # self.embeddings has embeddings model
        # self.metadata_db has metadata db

        # Get namespace from collection_id
        # Need to create OpenAIEmbeddings from chunks here too - which is currently in generate_report like this:

        # TODO: This config should be optional?
        embedding_creator = OpenAIEmbeddings(OpenAIEmbeddingsConfig())

        embeddings_list = await embedding_creator.transform_documents(documents)

        for idx_vectors_chunk in chunks(embeddings_list, batch_size=100):
            index.upsert(namespace="example-namespace", vectors=idx_vectors_chunk)

        # Use this for batching to pinecone
        # https://docs.pinecone.io/docs/insert-data#batching-upserts

        # OpenAIEmbeddings is doing things slowly though - will want to run batches by implementing transform_documents
        # instead of using embed directly (batching speeds up quite a bit)
        # Also pinecone upsert should also be using batching of X amount like in data_ingestion

        # Example on upsert from pinecone docs
        # upsert_response = index.upsert(
        #     namespace="example-namespace",
        #     vectors=[
        #         (
        #             "vec1",  # Vector ID
        #             [0.1, 0.2, 0.3, 0.4],  # Dense vector values
        #             {"genre": "drama"},  # Vector metadata
        #         ),
        #         ("vec2", [0.2, 0.3, 0.4, 0.5], {"genre": "action"}),
        #     ],
        # )

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
            # print(type(match), dir(match))
            return VectorEmbedding(
                vector=match.values[:10],
                metadata=match.metadata,
                text=match.metadata["text"],
            )

        return list(map(_response_record_to_vector_embedding, query_response.matches))
