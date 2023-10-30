from abc import ABC, abstractmethod
from pydantic.dataclasses import dataclass

from semantic_retrieval.common.base import Attributable
from semantic_retrieval.common.storage import BlobIdentifier
from enum import Enum
from typing import Any, Optional

from semantic_retrieval.common.types import Record


@dataclass
class RawDocumentChunk(ABC):
    content: str
    metadata: dict[Any, Any]


@dataclass
class RawDocument(ABC):
    url: str
    data_source: Any  # TODO: Update this to DataSource type when it is defined
    name: str
    mime_type: str
    hash: Optional[str]
    blob_id: Optional[BlobIdentifier]
    document_id: str
    collection_id: Optional[str]

    @abstractmethod
    async def get_content() -> str:
        pass

    @abstractmethod
    async def get_chunked_content() -> list[RawDocumentChunk]:
        pass


class DocumentFragmentType(Enum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    LIST = "list"
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    CODE = "code"
    QUOTE = "quote"


class DocumentFragment(Record, Attributable):
    fragment_id: str
    hash: Optional[str]
    blob_id: Optional[BlobIdentifier]
    fragment_type: DocumentFragmentType
    document_id: str
    previous_fragment: Optional["DocumentFragment"]
    next_fragment: Optional["DocumentFragment"]
    children: Optional[list["DocumentFragment"]]

    @abstractmethod
    async def get_content() -> str:
        pass

    @abstractmethod
    def serialize() -> str:
        pass


class Document(Record, Attributable):
    document_id: str
    collection_id: Optional[str]

    fragments: list[DocumentFragment]

    @abstractmethod
    def serialize(self) -> str:
        pass


@dataclass
class IngestedDocument(Document):
    raw_document: RawDocument
