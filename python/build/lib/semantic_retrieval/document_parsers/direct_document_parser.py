import uuid
from result import Err, Ok, Result
from semantic_retrieval.document.document import (
    DirectDocumentFragment,
    DocumentFragmentType,
    IngestedDocument,
    RawDocument,
)
from semantic_retrieval.document_parsers.document_parser import DocumentParser
from typing import List


class DirectDocumentParser(DocumentParser):
    async def parse(self, document: RawDocument) -> Result[IngestedDocument, str]:
        # Not using chunked content
        content = (await document.get_content()).unwrap()

        content_result = await document.get_content()
        match content_result:
            case Ok(content):
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

                return Ok(
                    IngestedDocument(
                        raw_document=document,
                        document_id=document_id,
                        collection_id=document.collection_id,
                        fragments=fragments,
                        metadata={},
                        attributes={},
                    )
                )
            case Err(err):
                return Err(err)
