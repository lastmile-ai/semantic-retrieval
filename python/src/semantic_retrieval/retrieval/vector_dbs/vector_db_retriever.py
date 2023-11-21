from typing import Generic, Optional

from semantic_retrieval.common.types import R
from semantic_retrieval.data_store.vector_dbs.vector_db import (
    VectorDB,
    VectorDBQuery,
)
from semantic_retrieval.document.metadata.document_metadata_db import (
    DocumentMetadataDB,
)
from semantic_retrieval.retrieval.document_retriever import DocumentRetriever
from semantic_retrieval.retrieval.retriever import BaseRetrieverQueryParams
from semantic_retrieval.utils.callbacks import CallbackManager


class VectorDBRetrieverQueryParams(BaseRetrieverQueryParams[VectorDBQuery]):
    query: VectorDBQuery


class BaseVectorDBRetriever(DocumentRetriever[R, VectorDBQuery], Generic[R]):
    vectorDB: VectorDB

    def __init__(
        self,
        vector_db: VectorDB,
        metadata_db: DocumentMetadataDB,
        callback_manager: Optional[CallbackManager] = None,
    ):
        super().__init__(metadata_db, callback_manager)
        self.vectorDB = vector_db

    async def getFragmentsUnsafe(self, params: VectorDBRetrieverQueryParams):
        # TODO [P1]
        pass

    async def retrieveData(self, params: VectorDBRetrieverQueryParams):
        # TODO [P1]
        pass
