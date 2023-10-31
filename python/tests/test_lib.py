import os
import pytest
from semantic_retrieval.document.metadata.in_memory_document_metadata_db import (
    InMemoryDocumentMetadataDB,
)
from semantic_retrieval.ingestion.data_sources.fs.file_system import FileSystem
from semantic_retrieval.document_parsers.multi_document_parser import (
    MultiDocumentParser,
    ParserConfig,
)
from semantic_retrieval.transformation.document.text.separator_text_chunker import (
    SeparatorTextChunker,
    SeparatorTextChunkerParams,
)
from semantic_retrieval.transformation.document.text.text_chunk_transformer import (
    TextChunkConfig,
)

metadata_db = InMemoryDocumentMetadataDB()


@pytest.mark.asyncio
async def test_create_index():
    rel_path_from_python_root = "examples/example_data/financial_report"
    cwd = os.path.normpath(os.getcwd())
    root_dir = os.path.join(cwd, "..") if cwd.endswith("python") else cwd
    full_path = os.path.join(root_dir, rel_path_from_python_root)
    file_system = FileSystem(full_path)
    raw_documents = file_system.load_documents()

    parsed_documents = await MultiDocumentParser().parse_documents(
        raw_documents,
        parser_config=ParserConfig(
            metadata_db=metadata_db, access_control_policy_factory=None
        ),
    )

    # print(raw_documents)
    # print(parsed_documents)

    documentTransformer = SeparatorTextChunker(
        SeparatorTextChunkerParams(
            metadata_db=metadata_db,
            text_chunk_config=TextChunkConfig(
                chunk_size_limit=500, chunk_overlap=100, size_fn=len
            ),
        )
    )

    # Transform the parsed documents
    await documentTransformer.transform_documents(parsed_documents)

    # TODO: Continue making stubs and essentially getting the demo as a test case (similar to localFileIngestion.ts right now)
    # Then can start to write the actual implementation / split the work

    assert False
