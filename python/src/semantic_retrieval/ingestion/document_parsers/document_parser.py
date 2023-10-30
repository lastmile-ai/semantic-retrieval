from typing import Any, Dict, List, Optional
from semantic_retrieval.common.base import Attributable

from semantic_retrieval.document.document import DocumentFragment, IngestedDocument, RawDocument
from semantic_retrieval.utils.callbacks import CallbackManager, Traceable


class DocumentParser(Attributable, Traceable):
    def __init__(self):
        self.mime_type_restriction: Optional[List[str]] = None

    def parse(self, raw_document: RawDocument) -> IngestedDocument:  # type: ignore [fixme]
        # TODO impl
        pass

    def parse_next(
        self,
        raw_document: RawDocument,
        previous_fragment: Optional[DocumentFragment] = None,
        take: Optional[int] = None,
    ) -> DocumentFragment:  # type: ignore [fixme]
        # TODO impl
        pass

    async def to_string(self, raw_document: RawDocument) -> str:  # type: ignore [fixme]
        # TODO impl
        pass

    async def serialize(self, raw_document: RawDocument) -> str:  # type: ignore [fixme]
        # TODO impl
        pass


class BaseDocumentParser(DocumentParser):
    def __init__(
        self,
        attributes: Dict[str, Any],
        metadata: Dict[str, Any],
        callback_manager: Optional[CallbackManager] = None,
    ):
        super().__init__()
        self.attributes = attributes if attributes else {}
        self.metadata = metadata if metadata else {}
        self.callback_manager = callback_manager

    def parse(self, raw_document: RawDocument) -> IngestedDocument: # type: ignore
        # TODO: impl
        pass

    def parse_next(
        self,
        raw_document: RawDocument,
        previous_fragment: Optional[DocumentFragment] = None,
        take: Optional[int] = None,
    ) -> DocumentFragment # type: ignore [fixme]
        # TODO impl
        pass

    async def to_string(self, raw_document: RawDocument) -> str:
        return await raw_document.get_content()

    async def serialize(self, raw_document: RawDocument) -> str:
        raise NotImplementedError("Method not implemented.")
