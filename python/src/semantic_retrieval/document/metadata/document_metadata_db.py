from typing import Any, Optional, List
from abc import ABC, abstractmethod

from semantic_retrieval.common.types import Record
from semantic_retrieval.document.metadata.document_metadata import DocumentMetadata


class DocumentMetadataQuery(Record):
    metadata_key: str
    metadata_value: Any
    match_type: str


class DocumentMetadataDB(ABC):
    # TODO: saqadri - implement a Postgres implementation of this interface.
    @abstractmethod
    async def get_metadata(self, document_id: str) -> Optional[DocumentMetadata]:
        pass

    @abstractmethod
    async def set_metadata(self, document_id: str, metadata: DocumentMetadata) -> None:
        pass

    @abstractmethod
    async def query_document_ids(self, query: DocumentMetadataQuery) -> List[str]:
        pass
