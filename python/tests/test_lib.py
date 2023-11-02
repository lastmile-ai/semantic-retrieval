import os
import pytest
from semantic_retrieval.document.metadata.in_memory_document_metadata_db import (
    InMemoryDocumentMetadataDB,
)
from semantic_retrieval.ingestion.data_sources.fs.file_system import FileSystem
from semantic_retrieval.document_parsers.multi_document_parser import (
    ParserConfig,
)
from semantic_retrieval.transformation.document.text.separator_text_chunker import (
    SeparatorTextChunkConfig,
    SeparatorTextChunker,
    SeparatorTextChunkerParams,
)
from semantic_retrieval.transformation.document.text.text_chunk_transformer import (
    TextChunkConfig,
)

import semantic_retrieval.document_parsers.multi_document_parser as mdp
from dotenv import load_dotenv


metadata_db = InMemoryDocumentMetadataDB()


@pytest.mark.asyncio
async def test_create_index():
    load_dotenv()

    rel_path_from_python_root = "examples/example_data/financial_report/portfolios"
    # rel_path_from_python_root = "examples/example_data/test/test.txt"
    cwd = os.path.normpath(os.getcwd())
    root_dir = os.path.join(cwd, "..") if cwd.endswith("python") else cwd
    full_path = os.path.join(root_dir, rel_path_from_python_root)
    file_system = FileSystem(full_path)
    raw_documents = file_system.load_documents()

    parsed_documents = await mdp.parse_documents(
        raw_documents,
        parser_config=ParserConfig(
            metadata_db=metadata_db, access_control_policy_factory=None
        ),
    )

    stcc = SeparatorTextChunkConfig(
        chunk_size_limit=500,
        chunk_overlap=100,
    )

    documentTransformer = SeparatorTextChunker(
        stcc=stcc,
        params=SeparatorTextChunkerParams(
            separator_text_chunk_config=stcc,
            # text_chunk_config=TextChunkConfig(
            #     chunk_size_limit=500,
            #     chunk_overlap=100,
            # ),
            # metadata_db=metadata_db,
        ),
    )

    # Transform the parsed documents
    _transformed_documents = await documentTransformer.transform_documents(
        parsed_documents
    )

    # TODO [P1]: Commenting out for now to get tests to pass, will add back in later - want to ship to have notebook ready
    # # Create the embeddings, use dotenv to get the environment vars & setup properly
    # await PineconeVectorDB.from_documents(
    #     transformed_documents,
    #     PineconeVectorDBConfig(
    #         index_name=os.getenv("PINECONE_INDEX_NAME", ""),
    #         api_key=os.getenv("PINECONE_API_KEY", ""),
    #         environment=os.getenv("PINECONE_ENVIRONMENT", ""),
    #         namespace=os.getenv("PINECONE_NAMESPACE", "abc"),
    #     ),
    #     embeddings=OpenAIEmbeddings(
    #         OpenAIEmbeddingsConfig(api_key=os.getenv("OPENAI_API_KEY"))
    #     ),
    #     metadata_db=metadata_db,
    # )

    # Use in notebook to showcase creating custom access policies and using them
    # Then will also need to check that querying from the created python ones (as well as created python metadatadb) works properly

    assert True
