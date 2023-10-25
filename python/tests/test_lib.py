import pytest
from semantic_retrieval.ingestion.data_sources.fs.file_system import FileSystem
from semantic_retrieval.document_parsers.multi_document_parser import (
    parse_documents,
    ParserConfig,
)

metadata_db = None  # TODO: Update to InMemoryMetadataDB once it is defined


@pytest.mark.asyncio
async def test_create_index():
    file_system = FileSystem("./example_docs")
    raw_documents = file_system.load_documents()

    await parse_documents(
        raw_documents,
        parser_config=ParserConfig(
            metadata_db=metadata_db, access_control_policy_factory=None
        ),
    )

    # TODO: Continue making stubs and essentially getting the demo as a test case (similar to localFileIngestion.ts right now)
    # Then can start to write the actual implementation / split the work

    assert True
