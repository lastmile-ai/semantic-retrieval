import uuid
from typing import Dict, List, Optional

from result import Err, Ok, Result
from semantic_retrieval.document.document import (
    DirectDocumentFragment,
    DocumentFragment,
    DocumentFragmentType,
    IngestedDocument,
    RawDocument,
)
from semantic_retrieval.ingestion.document_parsers.document_parser import (
    BaseDocumentParser,
)
from semantic_retrieval.utils.callbacks import CallbackManager, Traceable


class DirectDocumentParser(BaseDocumentParser, Traceable):
    def __init__(
        self,
        attributes: Dict[str, str],
        metadata: Optional[Dict[str, str]],
        callback_manager: Optional[CallbackManager] = None,
    ):
        super().__init__(attributes, metadata or {}, callback_manager)

    async def parse(
        self, raw_document: RawDocument
    ) -> Result[IngestedDocument, str]:
        # Not using chunked content
        content = (await raw_document.get_content()).unwrap()

        content_result = await raw_document.get_content()

        # TODO callbacks

        match content_result:
            case Ok(content):
                document_id = raw_document.document_id

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
                        raw_document=raw_document,
                        document_id=document_id,
                        collection_id=raw_document.collection_id,
                        fragments=fragments,
                        metadata={},
                        attributes={},
                    )
                )
            case Err(err):
                return Err(err)

    def parse_next(
        self,
        raw_document: RawDocument,
        previous_fragment: Optional[DocumentFragment] = None,
        take: Optional[int] = None,
    ) -> DocumentFragment:
        raise NotImplementedError()
