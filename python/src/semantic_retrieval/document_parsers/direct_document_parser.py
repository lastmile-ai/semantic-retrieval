import uuid
from semantic_retrieval.document.document import (
    DirectDocumentFragment,
    DocumentFragmentType,
    IngestedDocument,
    RawDocument,
)
from semantic_retrieval.document_parsers.document_parser import DocumentParser
from typing import List


class DirectDocumentParser(DocumentParser):
    async def parse(self, document: RawDocument) -> IngestedDocument:
        # Not using chunked content
        content = (await document.get_content()).unwrap()

        document_id = document.document_id

        # Only make one fragment with the entire contents of the document for now
        fragments: List[DirectDocumentFragment] = [
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
