from dataclasses import dataclass
from typing import Optional, Sequence

from semantic_retrieval.common.types import CallbackEvent
from semantic_retrieval.document.document import IngestedDocument, RawDocument
from semantic_retrieval.document.metadata.document_metadata import (
    DocumentMetadata,
)
from semantic_retrieval.document.metadata.document_metadata_db import (
    DocumentMetadataDB,
)
from semantic_retrieval.document_parsers.parser_registry import ParserRegistry
from semantic_retrieval.utils.callbacks import CallbackManager


@dataclass
class ParserConfig:
    metadata_db: Optional[DocumentMetadataDB]
    # access_control_policy_factory: Optional[
    #     DocumentAccessPolicyFactory
    # ]  # TODO [P1]: Add type for access control policy factory in followup diff
    parser_registry: Optional[ParserRegistry] = None


async def parse_documents(
    documents: Sequence[RawDocument],
    metadata_db: Optional[DocumentMetadataDB],
    callback_manager: CallbackManager,
) -> Sequence[IngestedDocument]:
    parser_registry = ParserRegistry()

    ingested_documents = []
    for document in documents:
        parser = parser_registry.get_parser(document.mime_type)
        ingested_document = (await parser.parse(document)).unwrap()

        if metadata_db is not None:
            # access_policies = []
            # if parser_config.access_control_policy_factory:
            #     access_policies = await parser_config.access_control_policy_factory.get_access_policies(
            #         document
            #     )

            await metadata_db.set_metadata(
                document.document_id,
                DocumentMetadata(
                    document_id=document.document_id,
                    uri=document.uri,
                    mime_type=document.mime_type,
                    metadata={},
                    attributes={},
                    # access_policies=access_policies,
                ),
            )

        ingested_documents.append(ingested_document)

    await callback_manager.run_callbacks(
        CallbackEvent(
            name="multi_parse_documents",
            data=dict(ingested_documents=ingested_documents),
        )
    )

    return ingested_documents
