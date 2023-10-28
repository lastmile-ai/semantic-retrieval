from abc import abstractmethod, ABC
from semantic_retrieval.document.metadata.document_metadata import DocumentMetadata


class DocumentMetadataDB(ABC):
    @abstractmethod
    async def get_metadata(self, document_id: str) -> DocumentMetadata | None:  # type: ignore
        pass

    @abstractmethod
    async def set_metadata(self, document_id: str, metadata: DocumentMetadata) -> None:  # type: ignore
        pass
