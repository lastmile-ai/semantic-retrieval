from abc import abstractmethod, ABC
from semantic_retrieval.document.metadata.document_metadata import DocumentMetadata
from typing import Awaitable


class DocumentMetadataDB(ABC):
    @abstractmethod
    async def get_metadata(self, documentId: str) -> Awaitable[DocumentMetadata]:  # type: ignore
        pass

    @abstractmethod
    async def set_metadata(self, documentId: str, metadata: DocumentMetadata) -> None:  # type: ignore
        pass
