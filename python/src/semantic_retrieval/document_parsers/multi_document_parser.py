from dataclasses import dataclass
from semantic_retrieval.document.document import RawDocument, IngestedDocument
from typing import Any, Optional, Sequence
from semantic_retrieval.document.metadata.document_metadata_db import DocumentMetadataDB
from semantic_retrieval.document_parsers.parser_registry import ParserRegistry

from semantic_retrieval.ingestion.document_parsers.document_parser import DocumentParser


@dataclass
class ParserConfig:
    metadata_db: DocumentMetadataDB
    access_control_policy_factory: Optional[
        Any
    ]  # TODO: Add type for access control policy factory in followup diff
    parser_registry: Optional[ParserRegistry] = None


class MultiDocumentParser(DocumentParser):
    async def parse_documents(
        self, documents: Sequence[RawDocument], parser_config: ParserConfig
    ) -> Sequence[IngestedDocument]:
        
        parser_registry = parser_config.parser_registry or ParserRegistry()

        parser_registry.get_parser(documents[0].mime_type)




        return []
