import asyncio
import logging
import sys
from typing import List
from semantic_retrieval.access_control.access_function import always_allow
from semantic_retrieval.access_control.access_identity import AuthenticatedIdentity

from semantic_retrieval.common.core import LOGGER_FMT, make_run_id
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

from semantic_retrieval.document_parsers.multi_document_parser import (
    ParserConfig,
)


from semantic_retrieval.access_control.always_allow_document_access_policy_factory import (
    AlwaysAllowDocumentAccessPolicyFactory,
)


from semantic_retrieval.transformation.document.text.separator_text_chunker import (
    SeparatorTextChunkConfig,
    SeparatorTextChunker,
    SeparatorTextChunkerParams,
)

from semantic_retrieval.data_store.vector_dbs.pinecone_vector_db import (
    PineconeVectorDB,
    PineconeVectorDBConfig,
)

from semantic_retrieval.transformation.embeddings.openai_embeddings import (
    OpenAIEmbeddings,
    OpenAIEmbeddingsConfig,
)

import semantic_retrieval.document_parsers.multi_document_parser as mdp
from semantic_retrieval.utils.callbacks import CallbackManager


logger = logging.getLogger(__name__)
logging.basicConfig(format=LOGGER_FMT)


async def main(argv: List[str]):
    loggers = [logger, pinecone_vector_db.logger]

    args = set_up_script(argv, loggers)
    config = get_config(args)
    logger.debug("CONFIG:\n")
    logger.debug(str(config))

    return await run_ingest(config)


async def run_ingest(config: Config):
    callback_manager = CallbackManager.default()
    run_id = make_run_id()
    # Create a new FileSystem instance
    fs_path = resolve_path(config.data_root, config.path_10ks)

    logger.info("Creating a new FileSystem instance")
    fileSystem = FileSystem(fs_path, callback_manager=callback_manager)

    # Load documents using the FileSystem instance
    logger.info("Loading documents")
    rawDocuments = await fileSystem.load_documents(run_id)
    logger.debug(f"RAW DOCUMENTS: {rawDocuments}")

    # Initialize an in-memory metadata DB
    logger.info("Initializing an in-memory metadata DB")
    metadata_db = InMemoryDocumentMetadataDB(callback_manager=CallbackManager.default())

    # Parse the raw documents
    logger.info("Parsing the raw documents")
    parsedDocuments = await mdp.parse_documents(
        rawDocuments,
        ParserConfig(
            metadata_db=metadata_db,
            access_control_policy_factory=AlwaysAllowDocumentAccessPolicyFactory(),
        ),
        callback_manager=callback_manager,
        run_id=run_id,
    )

    # Initialize a document transformer
    separator_text_chunk_config = SeparatorTextChunkConfig(
        chunk_size_limit=500,
        chunk_overlap=100,
    )

    # TODO [P1] set parameters better
    documentTransformer = SeparatorTextChunker(
        separator_text_chunk_config=separator_text_chunk_config,
        params=SeparatorTextChunkerParams(
            separator_text_chunk_config=separator_text_chunk_config,
        ),
        callback_manager=callback_manager,
    )

    # Transform the parsed documents
    logger.info("Transforming the parsed documents")
    transformedDocuments = await documentTransformer.transform_documents(
        parsedDocuments, run_id=run_id
    )

    # Generate a new namespace using UUID

    # Create a PineconeVectorDB instance and index the transformed documents
    pinecone_vectordb_config = PineconeVectorDBConfig(
        index_name=config.index_name,
        namespace=config.namespace,
        api_key=config.pinecone_key,
        environment=config.pinecone_environment,
    )

    openai_embedding_config = OpenAIEmbeddingsConfig(api_key=config.openai_key)

    embeddings = OpenAIEmbeddings(
        openai_embedding_config, callback_manager=callback_manager
    )

    logger.info("Creating a new PineconeVectorDB instance")
    _pineconeVectorDB = await PineconeVectorDB.from_documents(
        transformedDocuments,
        pinecone_vectordb_config,
        embeddings,
        metadata_db,
        # We will give permission to everything for the ingestion step.
        # We can use always_allow() and mock viewer identity for this.
        user_access_function=always_allow(),
        viewer_identity=AuthenticatedIdentity.mock(),
        callback_manager=callback_manager,
    )

    # TODO [P1]: validate state of pineconeVectorDB
    # print(f"{pineconeVectorDB=}")
    # print(f"{metadata_db.metadata.keys()=}")

    _metadata_persist_res = await metadata_db.persist(get_metadata_db_path(config))


if __name__ == "__main__":
    asyncio.run(main(sys.argv))
