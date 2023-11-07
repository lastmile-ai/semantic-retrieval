from abc import abstractmethod
from typing import TypeVar, Optional, List

from result import Result
from semantic_retrieval.document.document import Document, DocumentFragment

from semantic_retrieval.document.metadata.document_metadata_db import DocumentMetadataDB
from semantic_retrieval.retrieval.retriever import (
    BaseRetriever,
    BaseRetrieverQueryParams,
)
from semantic_retrieval.utils.callbacks import CallbackManager

R = TypeVar("R")
Q = TypeVar("Q")


class DocumentRetriever(BaseRetriever[R, Q]):
    def __init__(
        self,
        metadata_db: DocumentMetadataDB,
        callback_manager: Optional[CallbackManager] = None,
    ):
        super().__init__(metadata_db, callback_manager)
        self.metadata_db = metadata_db

    async def get_documents_for_fragments(self, fragments: List[DocumentFragment]):  # type: ignore [fixme]
        # TODO [P1] impl
        documents = []

        return documents

    @abstractmethod
    async def process_documents(self, _documents: List[Document]) -> R:  # type: ignore [fixme]
        pass

    @abstractmethod
    async def retrieve_data(
        self, params: BaseRetrieverQueryParams[Q]
    ) -> Result[R, str]:
        pass
