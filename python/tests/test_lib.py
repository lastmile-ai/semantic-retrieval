import os
import pytest
from semantic_retrieval.ingestion.data_sources.fs.file_system import FileSystem
from semantic_retrieval.document_parsers.multi_document_parser import (
    MultiDocumentParser,
    ParserConfig,
)
from semantic_retrieval.transformation.embeddings.openai_embeddings import (
    OpenAIEmbeddings,
    OpenAIEmbeddingsConfig,
)

metadata_db = None  # TODO: Update to InMemoryMetadataDB once it is defined


@pytest.mark.asyncio
async def test_create_index():
    rel_path_from_python_root = "examples/example_data/financial_report"
    cwd = os.path.normpath(os.getcwd())
    root_dir = os.path.join(cwd, "..") if cwd.endswith("python") else cwd
    full_path = os.path.join(root_dir, rel_path_from_python_root)
    file_system = FileSystem(full_path)
    raw_documents = file_system.load_documents()

    await MultiDocumentParser().parse_documents(
        raw_documents,
        parser_config=ParserConfig(
            metadata_db=metadata_db, access_control_policy_factory=None
        ),
    )

    # TODO: Continue making stubs and essentially getting the demo as a test case (similar to localFileIngestion.ts right now)
    # Then can start to write the actual implementation / split the work

    assert True


@pytest.mark.asyncio
async def test_openai_emb_query():
    cfg = OpenAIEmbeddingsConfig(
        api_key_path_abs="/Users/jonathan/keys/dev_OPENAI_API_KEY.txt"
    )
    e = OpenAIEmbeddings(cfg)
    res = await e.embed("hello world")
    dim = len(res.vector)
    assert dim == 1536
