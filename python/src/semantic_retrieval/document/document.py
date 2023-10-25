from abc import ABC, abstractmethod

from semantic_retrieval.common.base import Attributable
from semantic_retrieval.common.storage import BlobIdentifier
from enum import Enum


class RawDocumentChunk(ABC):
    content: str
    metadata: dict[any, any]


class RawDocument(ABC):
    url: str
    data_source: any  # TODO: Update this to DataSource type when it is defined
    name: str
    mime_type: str
    hash: str | None
    blob_id: BlobIdentifier | None
    document_id: str
    collection_id: str | None

    @abstractmethod
    def get_content() -> str:
        return "Not Implemented"

    @abstractmethod
    def get_chunked_content() -> list[RawDocumentChunk]:
        return []


class DocumentFragmentType(Enum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    LIST = "list"
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    CODE = "code"
    QUOTE = "quote"


class DocumentFragment(Attributable):
    fragment_id: str
    hash: str | None
    blob_id: BlobIdentifier | None
    fragment_type: DocumentFragmentType
    document_id: str
    previous_fragment: "DocumentFragment" | None
    next_fragment: "DocumentFragment" | None
    children: list["DocumentFragment"] | None

    @abstractmethod
    def get_content() -> str:
        return "Not Implemented"

    @abstractmethod
    def serialize() -> str:
        return "Not Implemented"


class Document(Attributable):
    document_id: str
    collection_id: str | None

    fragments: list[DocumentFragment]

    @abstractmethod
    def serialize(self) -> str:
        pass


class IngestedDocument(Document):
    raw_document: RawDocument
