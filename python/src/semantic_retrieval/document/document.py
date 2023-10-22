from abc import ABC, abstractmethod


class RawDocumentChunk(ABC):
    content: str
    metadata: dict[any, any]


class RawDocument(ABC):
    url: str
    name: str
    document_id: str

    @abstractmethod
    def get_content() -> str:
        return "Not Implemented"

    @abstractmethod
    def get_chunked_content() -> list[RawDocumentChunk]:
        return []
