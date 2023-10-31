import uuid
from semantic_retrieval.common.types import Record
from semantic_retrieval.document.document import (
    DocumentFragment,
    DocumentFragmentType,
    IngestedDocument,
    RawDocument,
)
from semantic_retrieval.document_parsers.document_parser import DocumentParser
from typing import List, Any, Optional


# TODO (suyog): Same issue as with RawDocument when converting from Typescript that happened in FileSystem
class DirectDocumentFragment(DocumentFragment, Record):
    content: str
    metadata: Optional[dict[Any, Any]]
    attributes: Optional[dict[Any, Any]]

    async def get_content(self) -> str:
        return self.content

    def serialize(self) -> str:
        return self.content


class DirectDocumentParser(DocumentParser):
    async def parse(self, document: RawDocument) -> IngestedDocument:
        # Not using chunked content
        content = (await document.get_content()).unwrap()

        document_id = document.document_id

        # Only make one fragment with the entire contents of the document for now
        fragments: List[DocumentFragment] = [
            DirectDocumentFragment(
                fragment_id=str(uuid.uuid4()),
                fragment_type=DocumentFragmentType.TEXT,
                document_id=document_id,
                content=content,
                hash=None,
                metadata={},
                attributes={},
            )
        ]

        return IngestedDocument(
            raw_document=document,
            document_id=document_id,
            collection_id=document.collection_id,
            fragments=fragments,
            metadata={},
            attributes={},
        )
