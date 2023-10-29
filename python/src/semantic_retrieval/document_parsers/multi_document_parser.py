from dataclasses import dataclass
from semantic_retrieval.access_control.document_access_policy_factory import DocumentAccessPolicyFactory
from semantic_retrieval.document.document import RawDocument, IngestedDocument
from typing import Optional, Sequence
from semantic_retrieval.document.metadata.document_metadata_db import DocumentMetadataDB
from semantic_retrieval.document_parsers.parser_registry import ParserRegistry

from semantic_retrieval.ingestion.document_parsers.document_parser import DocumentParser


@dataclass
class ParserConfig:
    metadata_db: DocumentMetadataDB
    access_control_policy_factory: Optional[
        DocumentAccessPolicyFactory
    ]  # TODO: Add type for access control policy factory in followup diff
    parser_registry: Optional[ParserRegistry] = None


class MultiDocumentParser(DocumentParser):
    async def parse_documents(
        self, documents: Sequence[RawDocument], parser_config: ParserConfig
    ) -> Sequence[IngestedDocument]:
        
        parser_registry = parser_config.parser_registry or ParserRegistry()

        ingested_documents = []
        for document in documents:
            parser = parser_registry.get_parser(document.mime_type)
            ingested_document = await parser.parse(document)
            ingested_documents.append(ingested_document)

        return ingested_documents
