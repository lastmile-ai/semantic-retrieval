from semantic_retrieval.document.document import IngestedDocument, RawDocument
from semantic_retrieval.document_parsers.document_parser import DocumentParser


class DirectDocumentParser(DocumentParser):
    async def parse(self, document: RawDocument) -> IngestedDocument:
        return IngestedDocument(
            raw_document=document,
            document_id=document.document_id,
            collection_id=document.collection_id,
            fragments=[],
        )
