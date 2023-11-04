import asyncio
import logging
import sys
from typing import List

from semantic_retrieval.common.core import LOGGER_FMT
from semantic_retrieval.examples.financial_report.config import (
    Config,
    get_config,
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


logger = logging.getLogger(__name__)
logging.basicConfig(format=LOGGER_FMT)


async def main(argv: List[str]):
    loggers = [logger]

    args = set_up_script(argv, loggers)
    config = get_config(args)
    logger.debug("CONFIG:\n")
    logger.debug(str(config))

    return await run_ingest(config)


async def run_ingest(config: Config):
    # Create a new FileSystem instance
    fileSystem = FileSystem(config.data_root)

    # Load documents using the FileSystem instance
    rawDocuments = fileSystem.load_documents()
    print(f"RAW DOCUMENTS: {rawDocuments}")

    # Initialize an in-memory metadata DB
    metadata_db = InMemoryDocumentMetadataDB()

    # Parse the raw documents
    parsedDocuments = await mdp.parse_documents(
        rawDocuments,
        ParserConfig(
            metadata_db=metadata_db,
            access_control_policy_factory=AlwaysAllowDocumentAccessPolicyFactory(),
        ),
    )

    # Initialize a document transformer
    # TODO [P1] set parameters better
    stcc = SeparatorTextChunkConfig(
        chunk_size_limit=500,
        chunk_overlap=100,
    )
    documentTransformer = SeparatorTextChunker(
        stcc=stcc,
        params=SeparatorTextChunkerParams(
            separator_text_chunk_config=stcc,
        ),
    )

    # Transform the parsed documents
    transformedDocuments = await documentTransformer.transform_documents(
        parsedDocuments
    )

    # Generate a new namespace using UUID
    namespace = "ns123"
    print(f"NAMESPACE: {namespace}")

    # Create a PineconeVectorDB instance and index the transformed documents
    pcvdbcfg = PineconeVectorDBConfig(
        index_name=config.index_name,
        namespace=config.namespace,
        api_key=config.pinecone_key,
        environment=config.pinecone_environment,
    )

    openaiembcfg = OpenAIEmbeddingsConfig(api_key=config.openai_key)

    embeddings = OpenAIEmbeddings(openaiembcfg)

    pineconeVectorDB = await PineconeVectorDB.from_documents(  # type: ignore [fixme TODO]
        transformedDocuments,
        pcvdbcfg,
        embeddings,
        metadata_db,
    )

    # TODO [P1]: validate state of pineconeVectorDB
    print(f"{pineconeVectorDB=}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv))
