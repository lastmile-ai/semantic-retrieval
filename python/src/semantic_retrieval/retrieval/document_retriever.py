from semantic_retrieval.retrieval.retriever import BaseRetriever
from semantic_retrieval.document.metadata.document_metadata_db import DocumentMetadataDB

from semantic_retrieval.utils.callbacks import (
    RetrieverProcessDocumentsEvent,
)


class DocumentRetriever(BaseRetriever):
    def __init__(self, metadata_db: DocumentMetadataDB):
        super().__init__(metadata_db)

    async def process_documents(self, documents):
        event = RetrieverProcessDocumentsEvent(
            name="onRetrieverProcessDocuments", documents=documents
        )
        self.callback_manager.run_callbacks(event)
        # TODO
