from typing import List, Optional

from semantic_retrieval.data_store.vector_dbs.vector_db import VectorDB
from semantic_retrieval.document.document import Document
from semantic_retrieval.document.metadata.document_metadata_db import (
    DocumentMetadataDB,
)
from semantic_retrieval.retrieval.vector_dbs.vector_db_retriever import (
    BaseVectorDBRetriever,
)
from semantic_retrieval.utils.callbacks import CallbackManager


class VectorDBDocumentRetriever(BaseVectorDBRetriever[List[Document]]):
    def __init__(
        self,
        vector_db: VectorDB,
        metadata_db: DocumentMetadataDB,
        callback_manager: Optional[CallbackManager] = None,
    ):
        super().__init__(vector_db, metadata_db, callback_manager)

    async def process_documents(
        self, documents: List[Document]
    ) -> List[Document]:
        return documents
