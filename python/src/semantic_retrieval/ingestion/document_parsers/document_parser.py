from typing import List, Optional

from semantic_retrieval.document.document import RawDocument, DocumentFragment, IngestedDocument

from semantic_retrieval.utils.callbacks import CallbackManager, Traceable

from semantic_retrieval.common.base import Attributable


class DocumentParser(Attributable, Traceable):
    def __init__(self, mimeType_restriction: List[Optional[str]] = None):
        self.mimeTypeRestriction = mimeType_restriction

    async def parse(self, raw_document: RawDocument) -> IngestedDocument:
        # Implement the parse logic for the entire document
        pass

    async def parse_next(
        self,
        raw_document: RawDocument,
        previous_fragment: DocumentFragment = None,
        take: int = None,
    ) -> DocumentFragment:
        # Implement the parse logic for the next fragment
        pass

    async def to_string(self, raw_document: RawDocument) -> str:
        return await raw_document.get_content()

    async def serialize(self, raw_document: RawDocument) -> str:
        # Implement the serialization logic
        pass


class BaseDocumentParser(DocumentParser):
    def __init__(self, attributes=None, metadata=None, callback_manager=None):
        super().__init__(attributes, metadata, callback_manager)

    async def parse(self, raw_document: RawDocument) -> IngestedDocument:
        # Implement the parsing logic for the entire document
        pass
