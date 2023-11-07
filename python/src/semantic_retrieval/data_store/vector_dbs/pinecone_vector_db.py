from functools import partial
from typing import Any, Dict, List

import pinecone
from pinecone import ScoredVector
from semantic_retrieval.access_control.access_function import (
    AccessFunction,
    get_data_access_checked_list,
)
from semantic_retrieval.access_control.access_identity import AuthenticatedIdentity
from semantic_retrieval.common.types import CallbackEvent, Record
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
from semantic_retrieval.utils.callbacks import CallbackManager, Traceable


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


class PineconeVectorDB(VectorDB, Traceable):
    def __init__(
        self,
        config: PineconeVectorDBConfig,
        embeddings: DocumentEmbeddingsTransformer,
        metadata_db: DocumentMetadataDB,
        user_access_function: AccessFunction,
        viewer_identity: AuthenticatedIdentity,
        callback_manager: CallbackManager,
    ):
        self.config = config
        self.embeddings = embeddings
        self.metadata_db = metadata_db
        self.user_access_function = user_access_function
        self.viewer_identity = viewer_identity
        self.callback_manager = callback_manager

    @classmethod
    async def from_documents(
        cls,
        documents: List[Document],
        config: PineconeVectorDBConfig,
        embeddings: DocumentEmbeddingsTransformer,
        metadata_db: DocumentMetadataDB,
        user_access_function: AccessFunction,
        viewer_identity: AuthenticatedIdentity,
        callback_manager: CallbackManager,
    ):
        instance = cls(
            config,
            embeddings,
            metadata_db,
            user_access_function,
            viewer_identity,
            callback_manager,
        )
        await instance.add_documents(documents)

        await callback_manager.run_callbacks(
            CallbackEvent(
                name="pinecone_vector_db_created",
                data=dict(
                    vector_db=instance,
                    config=config,
                    embeddings=embeddings,
                    metadata_db=metadata_db,
                    user_access_function=user_access_function,
                    viewer_identity=viewer_identity,
                    documents=documents,
                ),
            )
        )
        return instance

    def sanitize_metadata(self, unsanitized_metadata: Record):
        # TODO [P1]
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

        await self.callback_manager.run_callbacks(
            CallbackEvent(
                name="pinecone_vector_db_documents_added",
                data=dict(
                    vector_db=self,
                    documents=documents,
                    embeddings=embeddings_list,
                ),
            )
        )

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

        async def _resource_auth_id_fn(
            vector_embedding: VectorEmbedding,
        ) -> str:
            md = vector_embedding.metadata or {}
            doc_id = md.get("documentId", "")
            return doc_id

        query_res = await get_data_access_checked_list(
            query_params,
            self.user_access_function,
            _run_query,
            partial(_resource_auth_id_fn),
            self.viewer_identity.viewer_auth_id,
            cm=self.callback_manager,
        )

        await self.callback_manager.run_callbacks(
            CallbackEvent(
                name="pinecone_vector_db_query",
                data=dict(
                    vector_db=self,
                    query=query,
                    query_res=query_res,
                    params=query_params,
                ),
            )
        )

        return query_res


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
            vector=match.values,
            metadata=match.metadata,
            text=match.metadata["text"],
        )

    return list(map(_response_record_to_vector_embedding, query_response.matches))
