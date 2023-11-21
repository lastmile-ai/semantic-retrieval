import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Iterable, List
from uuid import uuid4

import pinecone
from pinecone import ScoredVector
from result import Err, Ok
from semantic_retrieval.access_control.access_function import (
    AccessFunction,
    get_data_access_checked_list,
)
from semantic_retrieval.access_control.access_identity import (
    AuthenticatedIdentity,
)
from semantic_retrieval.common.core import LOGGER_FMT, unflatten_iterable
from semantic_retrieval.common.types import CallbackEvent, Record
from semantic_retrieval.data_store.vector_dbs.vector_db import (
    VectorDB,
    VectorDBConfig,
    VectorDBEmbeddingQuery,
    VectorDBQuery,
    VectorDBTextQuery,
)
from semantic_retrieval.document.document import Document
from semantic_retrieval.document.metadata.document_metadata_db import (
    DocumentMetadataDB,
)
from semantic_retrieval.transformation.embeddings.embeddings import (
    DocumentEmbeddingsTransformer,
    VectorEmbedding,
)
from semantic_retrieval.transformation.embeddings.openai_embeddings import (
    OpenAIEmbeddingsHandle,
)
from semantic_retrieval.utils.callbacks import (
    CallbackManager,
    Traceable,
    safe_serialize_arbitrary_for_logging,
)
from semantic_retrieval.utils.interop import canonical_field

logger = logging.getLogger(__name__)
logging.basicConfig(format=LOGGER_FMT)


class PineconeVectorDBConfig(VectorDBConfig):
    index_name: str
    api_key: str
    environment: str
    namespace: str


UpsertResponseWrapper = Dict[str, Any]


@dataclass
class PCVector:
    id: str
    vector: List[float]
    metadata: Dict[str, Any]

    def as_tuple(self):
        return (self.id, self.vector, self.metadata)


@dataclass
class QueryParams:
    index: pinecone.Index
    namespace: str
    top_k: int
    metadata_filter: Dict[str, Any]
    vector: List[float]
    callback_manager: CallbackManager


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
    ) -> "PineconeVectorDB":
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
                    embeddings_transformer=embeddings,
                    metadata_db=metadata_db,
                    user_access_function=user_access_function,
                    viewer_identity=viewer_identity,
                    n_documents=len(documents),
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
        pinecone.init(
            api_key=self.config.api_key, environment=self.config.environment
        )

        embedding_creator = self.embeddings

        logger.debug("Getting embeddings")
        embeddings_list = await embedding_creator.transform_documents(
            documents
        )

        logger.info(f"Upserting {len(embeddings_list)} to Pinecone")

        def _ve_to_pcv(ve: VectorEmbedding, idx: int) -> PCVector:
            md = ve.metadata or {}
            # logger.debug(f'VE DOC ID={md.get("document_id", "no doc ID")=}')
            md_canonical = {canonical_field(f): v for f, v in md.items()}
            logger.debug(f"{md_canonical.keys()=}")
            return PCVector(
                id=uuid4().hex,
                vector=ve.vector,
                metadata=md_canonical,
            )

        _upsert_results = _batch_upsert(
            vectors_iterable=[
                _ve_to_pcv(ve, idx) for idx, ve in enumerate(embeddings_list)
            ],
            index_name=self.config.index_name,
            namespace=self.config.namespace,
            pool_threads=30,
            batch_size=100,
        )

        await self.callback_manager.run_callbacks(
            CallbackEvent(
                name="pinecone_vector_db_documents_added",
                data=dict(
                    vector_db=self,
                    n_documents=len(documents),
                    n_embeddings=len(embeddings_list),
                    n_upsert_batchs=len(_upsert_results),
                    first_upsert_res=_upsert_results[0],
                ),
            )
        )

        readys = [r["ready"] for r in _upsert_results]
        succesful = [r["successful"] for r in _upsert_results]
        if not all(readys):
            raise ValueError(
                "Something went horribly wrong: waited on all results, but not all ready."
            )

        n_failed = len([r for r in succesful if not r])
        if n_failed > 0:
            logger.error(f"Failed to upsert {n_failed} batches to Pinecone.")
            return Err(_upsert_results)
        else:
            logger.info("Upserted all embeddings to Pinecone.")
            return Ok(_upsert_results)

    async def query(self, query: VectorDBQuery) -> List[VectorEmbedding]:
        # TODO make the caller do embedding
        model_handle = OpenAIEmbeddingsHandle()

        async def _get_query_vector():
            match query:
                case VectorDBEmbeddingQuery(
                    embedding_vector=vec,
                ):
                    return vec
                case VectorDBTextQuery(text=text):
                    return await self.embeddings.embed(
                        text=text, model_handle=model_handle, metadata=None
                    )

        vec = await _get_query_vector()

        pinecone.init(
            api_key=self.config.api_key, environment=self.config.environment
        )
        index = pinecone.Index(self.config.index_name)

        top_k = query.topK
        metadata_filter = query.metadata_filter

        query_params = QueryParams(
            index=index,
            namespace=self.config.namespace,
            top_k=top_k,
            metadata_filter=metadata_filter,
            vector=vec.vector,
            callback_manager=self.callback_manager,
        )

        async def _resource_auth_id_fn(
            vector_embedding: VectorEmbedding,
        ) -> str:
            md = vector_embedding.metadata or {}
            doc_id = md.get(canonical_field("document_id"), "")
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
                name="pinecone_vector_db_query_post_check",
                data=dict(
                    vector_db=self,
                    query=query,
                    # query_res=query_res,
                    params=query_params,
                    n_res=len(query_res),
                ),
            )
        )

        return query_res


async def _run_query(query_params: QueryParams) -> List[VectorEmbedding]:
    query_response = query_params.index.query(
        namespace=query_params.namespace,
        top_k=query_params.top_k,
        include_values=True,
        include_metadata=True,
        vector=query_params.vector,
        filter=query_params.metadata_filter,
    )

    await query_params.callback_manager.run_callbacks(
        CallbackEvent(
            name="pinecone_vector_db_query_pre_check",
            data=dict(
                params=query_params,
                n_res=len(query_response.matches),
            ),
        )
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

    return list(
        map(_response_record_to_vector_embedding, query_response.matches)
    )


def _batch_upsert(
    vectors_iterable: Iterable[PCVector],
    index_name: str,
    namespace: str,
    pool_threads: int,
    batch_size: int,
) -> List[UpsertResponseWrapper]:
    with pinecone.Index(index_name, pool_threads=pool_threads) as index:
        logger.debug(f"[batch upsert] {index_name=}")
        the_vectors = list(vectors_iterable)
        safe_serialize_arbitrary_for_logging({"the_vectors": the_vectors})
        # Send requests in parallel
        async_results = [
            index.upsert(
                vectors=[v.as_tuple() for v in ids_vectors_chunk],
                async_req=True,
                namespace=namespace,
            )
            for ids_vectors_chunk in unflatten_iterable(
                the_vectors, chunk_size=batch_size
            )
        ]

        def _get_result(
            raw_result: pinecone.UpsertResponse,
        ) -> UpsertResponseWrapper:
            out = {}
            for k in dir(raw_result):
                out[k] = getattr(raw_result, k)
                # out["get"] = raw_result.get()
                out["wait"] = raw_result.wait()
                out["ready"] = raw_result.ready()
                out["successful"] = raw_result.successful()
            return out

        results = list(map(_get_result, async_results))
        return results
