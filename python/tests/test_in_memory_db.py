import pytest
from semantic_retrieval.document.metadata.document_metadata import DocumentMetadata
from semantic_retrieval.document.metadata.in_memory_document_metadata_db import (
    InMemoryDocumentMetadataDB,
)
import tempfile


@pytest.mark.asyncio
async def test_in_memory_db():
    db = InMemoryDocumentMetadataDB({})

    await db.set_metadata(
        "test",
        DocumentMetadata(document_id="test", uri="blah", metadata={}, attributes={}),
    )

    result = await db.get_metadata("test")
    assert result is not None
    assert result.document_id == "test"

    # Want to write to file, read from file & also delete that file too (python has tmp files)
    test_file = tempfile.NamedTemporaryFile(delete=True)
    db.persist(test_file.name)

    db2 = await InMemoryDocumentMetadataDB.from_json_file(test_file.name)
    result = await db2.get_metadata("test")
    assert result is not None
    assert result.document_id == "test"

    # with open(test_file.name, 'r') as f:
