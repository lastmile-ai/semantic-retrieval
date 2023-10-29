import pytest
from semantic_retrieval.document.metadata.in_memory_document_metadata_db import InMemoryDocumentMetadataDB
from semantic_retrieval.ingestion.data_sources.fs.file_system import FileSystem
from semantic_retrieval.document_parsers.multi_document_parser import (
    MultiDocumentParser,
    ParserConfig,
)

metadata_db = InMemoryDocumentMetadataDB()


@pytest.mark.asyncio
async def test_create_index():
    file_system = FileSystem("src/semantic_retrieval/examples/financial_report")
    raw_documents = file_system.load_documents()

    await MultiDocumentParser().parse_documents(
        raw_documents,
        parser_config=ParserConfig(metadata_db=metadata_db, access_control_policy_factory=None),
    )

    # TODO: Continue making stubs and essentially getting the demo as a test case (similar to localFileIngestion.ts right now)
    # Then can start to write the actual implementation / split the work

    assert True
