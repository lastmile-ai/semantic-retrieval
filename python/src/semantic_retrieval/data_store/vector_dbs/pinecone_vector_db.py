from functools import partial
import re
from typing import Any, Dict, List

import pinecone
from pinecone import ScoredVector
from semantic_retrieval.access_control.access_function import (
    AccessFunction,
    get_data_access_checked_list,
)
from semantic_retrieval.access_control.access_identity import AuthenticatedIdentity
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


class QueryParams:
    index: pinecone.Index
    namespace: str
    top_k: int
    metadata_filter: Dict[str, Any]
    vector: List[float]

    def __init__(
        self,
        index: pinecone.Index,
        namespace: str,
        top_k: int,
        metadata_filter: Dict[str, Any],
        vector: list[float],
    ) -> None:
        self.index = index
        self.namespace = namespace
        self.top_k = top_k
        self.metadata_filter = metadata_filter
        self.vector = vector


class PineconeVectorDB(VectorDB):
    def __init__(
        self,
        config: PineconeVectorDBConfig,
        embeddings: DocumentEmbeddingsTransformer,
        metadata_db: DocumentMetadataDB,
        user_access_function: AccessFunction,
        viewer_identity: AuthenticatedIdentity,
    ):
        self.config = config
        self.embeddings = embeddings
        self.metadata_db = metadata_db
        self.user_access_function = user_access_function
        self.viewer_identity = viewer_identity

    @classmethod
    async def from_documents(
        cls,
        documents: List[Document],
        config: PineconeVectorDBConfig,
        embeddings: DocumentEmbeddingsTransformer,
        metadata_db: DocumentMetadataDB,
        user_access_function: AccessFunction,
        viewer_identity: AuthenticatedIdentity,
    ):
        instance = cls(
            config, embeddings, metadata_db, user_access_function, viewer_identity
        )
        await instance.add_documents(documents)
        return instance

    def sanitize_metadata(self, unsanitized_metadata: Record):
        # TODO [P0]
        pass

    async def add_documents(
        self,
        documents: List[Document],
    ):
        pinecone.init(api_key=self.config.api_key, environment=self.config.environment)
        index = pinecone.Index(self.config.index_name)

        embedding_creator = self.embeddings

        embeddings_list = await embedding_creator.transform_documents(documents)

        # TODO [P0]: Update this to batch to get faster performance
        # Use this for batching to pinecone
        # https://docs.pinecone.io/docs/insert-data#batching-upserts
        for idx, embedding in enumerate(embeddings_list):
            metadata = {"text": embedding.text}
            vectors_chunk = embedding.vector
            index.upsert(namespace=self.config.namespace, vectors=[(f"vec{idx}", vectors_chunk, metadata)])  # type: ignore

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

        query_params = QueryParams(
            index=index,
            namespace=self.config.namespace,
            top_k=top_k,
            metadata_filter=metadata_filter,
            vector=vec.vector,
        )

        return await get_data_access_checked_list(
            query_params,
            self.user_access_function,
            _run_query,
            partial(_resource_auth_id_fn, metdata_db=self.metadata_db),
            self.viewer_identity.viewer_auth_id,
        )


async def _resource_auth_id_fn(
    vector_embedding: VectorEmbedding, metdata_db: DocumentMetadataDB
) -> str:
    """
    This takes a single datum (in this case vector embedding)
    and returns the ID which is passed to the user access control function.
    The ticker stands in for the auth ID that we have agreed on with the library users.

    That is, the user will take pairs of (ticker, viewer_auth_id).

    In production, this function would not have to do the metadata DB lookup.
    Instead we would have ingested the ticker directly with the vector metadata.

    """
    md = vector_embedding.metadata or {}
    doc_id = md.get("documentId", "")

    # print(f"{doc_id=}")
    metadata = await metdata_db.get_metadata(doc_id)
    # print(f"{metadata=}")
    uri = metadata.unwrap().uri
    # print(f"{uri=}")

    ticker_re = re.search(r".*_([A-Z\.]+)\..*", uri)
    out = ticker_re.groups()[0] if ticker_re else ""
    # print(f"{out=}")
    return out


def _run_query(query_params: QueryParams) -> List[VectorEmbedding]:
    query_response = query_params.index.query(
        namespace=query_params.namespace,
        top_k=query_params.top_k,
        include_values=True,
        include_metadata=True,
        vector=query_params.vector,
        filter=query_params.metadata_filter,
    )

    # TODO [P1] type better
    def _response_record_to_vector_embedding(
        match: ScoredVector,
    ) -> VectorEmbedding:
        return VectorEmbedding(
            # TODO [P1] ??
            vector=match.values[:10],
            metadata=match.metadata,
            text=match.metadata["text"],
        )

    return list(map(_response_record_to_vector_embedding, query_response.matches))
