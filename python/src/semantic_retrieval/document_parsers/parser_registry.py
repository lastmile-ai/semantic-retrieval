from semantic_retrieval.document_parsers.direct_document_parser import (
    DirectDocumentParser,
)
from semantic_retrieval.ingestion.document_parsers.document_parser import (
    DocumentParser,
)


class ParserRegistry:
    default_parser: DocumentParser
    parsers: dict[str, DocumentParser] = {}

    def __init__(self, default_parser: DocumentParser | None = None) -> None:
        if default_parser is not None:
            self.default_parser = default_parser
        else:
            self.default_parser = DirectDocumentParser(
                attributes={}, metadata={}
            )

    def register_parser(self, mime_type: str, parser: DocumentParser) -> None:
        self.parsers[mime_type] = parser

    def get_parser(self, mime_type: str) -> DocumentParser:
        return self.parsers.get(mime_type, self.default_parser)
