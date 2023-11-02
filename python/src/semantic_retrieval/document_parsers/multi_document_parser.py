from dataclasses import dataclass
from semantic_retrieval.access_control.document_access_policy_factory import (
    DocumentAccessPolicyFactory,
)
from semantic_retrieval.document.document import RawDocument, IngestedDocument
from typing import Optional, Sequence
from semantic_retrieval.document.metadata.document_metadata import DocumentMetadata
from semantic_retrieval.document.metadata.document_metadata_db import DocumentMetadataDB
from semantic_retrieval.document_parsers.parser_registry import ParserRegistry


@dataclass
class ParserConfig:
    metadata_db: Optional[DocumentMetadataDB]
    access_control_policy_factory: Optional[
        DocumentAccessPolicyFactory
    ]  # TODO [P1]: Add type for access control policy factory in followup diff
    parser_registry: Optional[ParserRegistry] = None


async def parse_documents(
        documents: Sequence[RawDocument], parser_config: ParserConfig
    ) -> Sequence[IngestedDocument]:
    parser_registry = parser_config.parser_registry or ParserRegistry()

    ingested_documents = []
    for document in documents:
        parser = parser_registry.get_parser(document.mime_type)
        ingested_document = (
            await parser.parse(document)
        ).unwrap()

        if parser_config.metadata_db is not None:
            access_policies = []
            if parser_config.access_control_policy_factory:
                access_policies = await parser_config.access_control_policy_factory.get_access_policies(
                    document
                )

            await parser_config.metadata_db.set_metadata(
                document.document_id,
                DocumentMetadata(
                    # TODO [P0]: These were removed, but may need to be added back
                    # document=ingested_document,
                    # raw_document=document,
                    document_id=document.document_id,
                    uri=document.uri,
                    mime_type=document.mime_type,
                    metadata={},
                    attributes={},
                    access_policies=access_policies,
                ),
            )

        ingested_documents.append(ingested_document)

    return ingested_documents
