import asyncio
import logging
import sys
from typing import List
from semantic_retrieval.access_control.access_function import AccessFunction, always_allow
from semantic_retrieval.access_control.access_identity import AuthenticatedIdentity

from semantic_retrieval.common.core import LOGGER_FMT
from semantic_retrieval.data_store.vector_dbs import pinecone_vector_db
from semantic_retrieval.examples.financial_report.config import (
    Config,
    get_config,
    get_metadata_db_path,
    resolve_path,
    set_up_script,
)

from semantic_retrieval.ingestion.data_sources.fs.file_system import FileSystem
from semantic_retrieval.document.metadata.in_memory_document_metadata_db import (
    InMemoryDocumentMetadataDB,
)


from semantic_retrieval.transformation.document.text.separator_text_chunker import (
    SeparatorTextChunker,
    SeparatorTextChunkerParams,
)

from semantic_retrieval.data_store.vector_dbs.pinecone_vector_db import (
    PineconeVectorDB,
    PineconeVectorDBConfig,
)
from semantic_retrieval.transformation.embeddings import openai_embeddings

from semantic_retrieval.transformation.embeddings.openai_embeddings import (
    OpenAIEmbeddings,
    OpenAIEmbeddingsConfig,
)

import semantic_retrieval.document_parsers.multi_document_parser as mdp
from semantic_retrieval.utils.callbacks import CallbackManager
from semantic_retrieval.utils import callbacks as lib_callbacks


logger = logging.getLogger(__name__)
logging.basicConfig(format=LOGGER_FMT)


async def main(argv: List[str]):
    loggers = [logger, pinecone_vector_db.logger, openai_embeddings.logger]

    args = set_up_script(argv, loggers)
    config = get_config(args)
    logger.debug("CONFIG:\n")
    logger.debug(str(config))

    return await run_ingest(config)


async def run_ingest(config: Config):
    # Make callback manager that writes all events to local JSON file
    callback_manager = lib_callbacks.CallbackManager(
        [
            lib_callbacks.to_json(
                "examples/example_data/financial_report/artifacts/callback_data.json"
            )
        ]
    )

    # We use `mock` identity here because we need ingestion
    # to have access to all data.
    viewer_identity = AuthenticatedIdentity.mock()
    access_function: AccessFunction = always_allow()

    # Initialize an in-memory metadata DB
    logger.info("Initializing an in-memory metadata DB")
    # We'll write this to disk later for use in generation.
    metadata_db_path = get_metadata_db_path(config)
    metadata_db = InMemoryDocumentMetadataDB(callback_manager=CallbackManager.default())

    # Create a new FileSystem instance
    fs_path = resolve_path(config.data_root, config.path_10ks)
    file_system = FileSystem(fs_path, callback_manager=callback_manager)

    # Load documents using the FileSystem instance
    raw_documents = await file_system.load_documents()

    # Configure embeddings for representing the ingested chunks
    openai_embedding_config = OpenAIEmbeddingsConfig(api_key=config.openai_key)
    embeddings = OpenAIEmbeddings(openai_embedding_config, callback_manager=callback_manager)

    # Create a PineconeVectorDB instance and index the transformed documents
    pinecone_vectordb_config = PineconeVectorDBConfig(
        index_name=config.index_name,
        namespace=config.namespace,
        api_key=config.pinecone_key,
        environment=config.pinecone_environment,
    )

    # Parse the raw documents
    parsed_documents = await mdp.parse_documents(
        raw_documents,
        metadata_db,
        callback_manager=callback_manager,
    )

    # Initialize a document transformer
    document_transformer = SeparatorTextChunker(
        params=SeparatorTextChunkerParams(
            separator=" ",
            strip_new_lines=True,
            chunk_size_limit=500,
            chunk_overlap=100,
            document_metadata_db=metadata_db,
        ),
        callback_manager=callback_manager,
    )

    # Transform the parsed documents
    transformed_documents = await document_transformer.transform_documents(
        parsed_documents,
    )

    # Upload the transformed documents to the vector store.
    _ = await PineconeVectorDB.from_documents(
        transformed_documents,
        pinecone_vectordb_config,
        embeddings,
        metadata_db,
        user_access_function=access_function,
        viewer_identity=viewer_identity,
        callback_manager=callback_manager,
    )

    # Write out the in-memory metadata DB to local disk for use in generation.
    # In production, this would be unnecessary.
    logger.info("Persisting metadata DB: " + metadata_db_path)
    _ = await metadata_db.persist(metadata_db_path)


if __name__ == "__main__":
    asyncio.run(main(sys.argv))
