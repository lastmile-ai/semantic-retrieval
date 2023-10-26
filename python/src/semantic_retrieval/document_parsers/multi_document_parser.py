from semantic_retrieval.document.document import RawDocument, IngestedDocument
from typing import Any, List, Optional, Sequence

from semantic_retrieval.ingestion.document_parsers.document_parser import DocumentParser


class ParserConfig:
    metadata_db: Optional[Any]  # TODO: Add type for metadata db in followup diff
    access_control_policy_factory: Optional[
        Any
    ]  # TODO: Add type for access control policy factory in followup diff
    parser_registry: Optional[Any]  # TODO: Add type for parser registry in followup diff

    def __init__(self, metadata_db=None, access_control_policy_factory=None, parser_registry=None):
        self.metadata_db = metadata_db
        self.access_control_policy_factory = access_control_policy_factory
        self.parser_registry = parser_registry


class MultiDocumentParser(DocumentParser):
    async def parse_documents(
        self, documents: Sequence[RawDocument], parser_config: ParserConfig
    ) -> Sequence[IngestedDocument]:
        return documents
