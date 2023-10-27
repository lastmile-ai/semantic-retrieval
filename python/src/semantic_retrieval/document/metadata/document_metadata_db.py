from abc import abstractmethod, ABC
from semantic_retrieval.document.metadata.document_metadata import DocumentMetadata


class DocumentMetadataDB(ABC):
    @abstractmethod
    async def getMetadata(self, documentId: str) -> DocumentMetadata:  # type: ignore
        pass

    @abstractmethod
    async def setMetadata(self, documentId: str, metadata: DocumentMetadata) -> None:  # type: ignore
        pass
