import pytest
from semantic_retrieval.document.metadata.document_metadata import DocumentMetadata
from semantic_retrieval.document.metadata.in_memory_document_metadata_db import (
    InMemoryDocumentMetadataDB,
)
import tempfile

from semantic_retrieval.utils.callbacks import CallbackManager


@pytest.mark.asyncio
async def test_in_memory_db():
    db = InMemoryDocumentMetadataDB(callback_manager=CallbackManager.default())

    await db.set_metadata(
        "test",
        DocumentMetadata(document_id="test", uri="blah", metadata={}, attributes={}),
    )

    result = await db.get_metadata("test")
    assert result.unwrap().document_id == "test"

    # Want to write to file, read from file & also delete that file too (python has tmp files)
    test_file = tempfile.NamedTemporaryFile(delete=True)
    _ = await db.persist(test_file.name)

    db2 = await InMemoryDocumentMetadataDB.from_json_file(test_file.name)
    d_id = db2.unwrap().metadata["test"].document_id
    assert d_id == "test"
