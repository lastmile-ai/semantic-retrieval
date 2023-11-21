from abc import abstractmethod
from typing import List, Optional, TypeVar

from result import Result
from semantic_retrieval.document.document import Document, DocumentFragment
from semantic_retrieval.document.metadata.document_metadata_db import (
    DocumentMetadataDB,
)
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

    async def get_documents_for_fragments(
        self, fragments: List[DocumentFragment]
    ) -> List[Document]:
        raise NotImplementedError()

    @abstractmethod
    async def process_documents(self, documents: List[Document]) -> R:
        pass

    @abstractmethod
    async def retrieve_data(
        self, params: BaseRetrieverQueryParams[Q]
    ) -> Result[R, str]:
        pass
