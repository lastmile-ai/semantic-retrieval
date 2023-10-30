from typing import Generic, Optional
from semantic_retrieval.common.types import R, Record
from semantic_retrieval.data_store.vector_dbs.vector_db import VectorDB, VectorDBQuery
from semantic_retrieval.document.metadata.document_metadata_db import DocumentMetadataDB
from semantic_retrieval.retrieval.document_retriever import DocumentRetriever
from semantic_retrieval.retrieval.retriever import BaseRetrieverQueryParams
from semantic_retrieval.utils.callbacks import CallbackManager


class VectorDBRetrieverParams(Record):
    vector_db: VectorDB
    metadata_db: DocumentMetadataDB
    callback_manager: Optional[CallbackManager] = None


class VectorDBRetrieverQueryParams(BaseRetrieverQueryParams[VectorDBQuery]):
    query: VectorDBQuery


class BaseVectorDBRetriever(DocumentRetriever[R, VectorDBQuery], Generic[R]):
    vectorDB: VectorDB

    def __init__(self, params: VectorDBRetrieverParams):
        super().__init__(params.metadata_db)
        self.vectorDB = params.vector_db

    async def getFragmentsUnsafe(self, params: VectorDBRetrieverQueryParams):
        # TODO
        pass

    async def retrieveData(self, params: VectorDBRetrieverQueryParams):
        # TODO
        pass
