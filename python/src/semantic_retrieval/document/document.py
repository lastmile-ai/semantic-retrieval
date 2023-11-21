from abc import abstractmethod
from enum import Enum
from typing import Any, List, Optional

from result import Result
from semantic_retrieval.common.base import Attributable
from semantic_retrieval.common.storage import BlobIdentifier
from semantic_retrieval.common.types import Record


class RawDocumentChunk(Record):
    content: str
    metadata: dict[Any, Any]


class RawDocument(Record):
    uri: str
    # data_source: Any  # TODO [P1]: Update this to DataSource type when it is defined
    name: str
    document_id: str
    hash: Optional[str]
    collection_id: Optional[str]
    mime_type: str = "unknown"
    blob_id: Optional[BlobIdentifier] = None

    @abstractmethod
    async def get_content(self) -> Result[str, str]:
        pass

    @abstractmethod
    async def get_chunked_content(self) -> List[RawDocumentChunk]:
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


class DocumentFragment(Attributable):
    fragment_id: str
    hash: Optional[str]
    fragment_type: DocumentFragmentType
    document_id: str
    previous_fragment: Optional["DocumentFragment"] = None
    next_fragment: Optional["DocumentFragment"] = None
    children: Optional[list["DocumentFragment"]] = None
    blob_id: Optional[BlobIdentifier] = None

    @abstractmethod
    async def get_content(self) -> str:
        pass

    @abstractmethod
    def serialize(self) -> str:
        pass


# TODO [P1] (suyog): Same issue as with RawDocument when converting from Typescript that happened in FileSystem
class DirectDocumentFragment(DocumentFragment, Record):
    content: str
    metadata: Optional[dict[Any, Any]]
    attributes: Optional[dict[Any, Any]]

    async def get_content(self) -> str:
        return self.content

    def serialize(self) -> str:
        return self.content


class Document(Attributable, Record):
    document_id: str
    collection_id: Optional[str]
    fragments: List[DirectDocumentFragment]

    @abstractmethod
    def serialize(self) -> str:
        pass


class IngestedDocument(Document):
    raw_document: RawDocument
    # Pydantic & pylance issues w/ not being able to use Attributable fields
    metadata: Optional[dict[Any, Any]]
    attributes: Optional[dict[Any, Any]]

    def serialize(self) -> str:
        return "Not Implemented"


class TransformedDocument(Document):
    document: Document
    # Pydantic & pylance issues w/ not being able to use Attributable fields
    metadata: Optional[dict[Any, Any]]
    attributes: Optional[dict[Any, Any]]

    def serialize(self) -> str:
        return "Not Implemented"
