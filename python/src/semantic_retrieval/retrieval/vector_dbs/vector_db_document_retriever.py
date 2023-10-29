from typing import Generic, List, TypeVar
from document.document import Document
from semantic_retrieval.retrieval.vector_dbs.vector_db_retriever import (
    BaseVectorDBRetriever,
    VectorDBRetrieverParams,
)
from utils.callbacks import RetrieverProcessDocumentsEvent

T = TypeVar("T")


class VectorDBDocumentRetriever(BaseVectorDBRetriever[List[Document]], Generic[T]):
    def __init__(self, params: VectorDBRetrieverParams):
        super().__init__(params)

    async def process_documents(self, documents: List[Document]) -> List[Document]:
        event = RetrieverProcessDocumentsEvent(
            name="onRetrieverProcessDocuments", documents=documents
        )
        if self.callback_manager:
            await self.callback_manager.run_callbacks(event)

        return documents
