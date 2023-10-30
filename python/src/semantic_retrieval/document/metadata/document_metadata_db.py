from abc import abstractmethod, ABC
from typing import Optional
from semantic_retrieval.document.metadata.document_metadata import DocumentMetadata


class DocumentMetadataDB(ABC):
    @abstractmethod
    async def get_metadata(self, document_id: str) -> Optional[DocumentMetadata]:
        pass

    @abstractmethod
    async def set_metadata(self, document_id: str, metadata: DocumentMetadata) -> None:
        pass
