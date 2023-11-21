from abc import abstractmethod
from typing import Any, Dict, List, Optional

from result import Result
from semantic_retrieval.common.base import Attributable
from semantic_retrieval.document.document import (
    DocumentFragment,
    IngestedDocument,
    RawDocument,
)
from semantic_retrieval.utils.callbacks import CallbackManager, Traceable


class DocumentParser(Attributable, Traceable):
    mime_type_restriction: Optional[List[str]] = None

    @abstractmethod
    async def parse(
        self, raw_document: RawDocument
    ) -> Result[IngestedDocument, str]:
        pass

    @abstractmethod
    def parse_next(
        self,
        raw_document: RawDocument,
        previous_fragment: Optional[DocumentFragment] = None,
        take: Optional[int] = None,
    ) -> DocumentFragment:
        pass

    @abstractmethod
    async def to_string(self, raw_document: RawDocument) -> str:
        pass

    @abstractmethod
    async def serialize(self, raw_document: RawDocument) -> str:
        pass


class BaseDocumentParser(DocumentParser):
    def __init__(
        self,
        attributes: Dict[str, Any],
        metadata: Dict[str, Any],
        callback_manager: Optional[CallbackManager] = None,
    ):
        super().__init__()
        # self.attributes = attributes if attributes else {}
        # self.metadata = metadata if metadata else {}
        # self.callback_manager = callback_manager

    @abstractmethod
    async def parse(
        self, raw_document: RawDocument
    ) -> Result[IngestedDocument, str]:
        pass

    @abstractmethod
    def parse_next(
        self,
        raw_document: RawDocument,
        previous_fragment: Optional[DocumentFragment] = None,
        take: Optional[int] = None,
    ) -> DocumentFragment:
        pass

    async def to_string(self, raw_document: RawDocument) -> str:
        content = await raw_document.get_content()
        return content.unwrap_or_else(str)

    async def serialize(self, raw_document: RawDocument) -> str:
        raise NotImplementedError("Method not implemented.")
