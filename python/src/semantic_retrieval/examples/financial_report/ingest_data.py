import asyncio
import sys
from typing import List
from semantic_retrieval.examples.financial_report.config import Config

from semantic_retrieval.ingestion.data_sources.fs.file_system import FileSystem
from semantic_retrieval.document.metadata.in_memory_document_metadata_db import (
    InMemoryDocumentMetadataDB,
)

from semantic_retrieval.document_parsers.multi_document_parser import (
    MultiDocumentParser,
    ParserConfig,
)


from semantic_retrieval.access_control.always_allow_document_access_policy_factory import (
    AlwaysAllowDocumentAccessPolicyFactory,
)


from semantic_retrieval.transformation.document.text.separator_text_chunker import (
    SeparatorTextChunker,
    SeparatorTextChunkerParams,
)

from semantic_retrieval.data_store.vector_dbs.pinecone_vector_db import (
    PineconeVectorDB,
    PineconeVectorDBConfig,
)
from semantic_retrieval.transformation.document.text.text_chunk_transformer import (
    TextChunkConfig,
)

from semantic_retrieval.transformation.embeddings.openai_embeddings import (
    OpenAIEmbeddings,
    OpenAIEmbeddingsConfig,
)

import argparse

from semantic_retrieval.utils.configs.configs import remove_nones


async def main(argv: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str)
    args = parser.parse_args(argv[1:])
    args_resolved = Config(**remove_nones(vars(args)))
    return await run_ingest(args_resolved)


async def run_ingest(config: Config):
    print(f"{config=}")
    # Create a new FileSystem instance
    fileSystem = FileSystem(config.data_root)

    # Load documents using the FileSystem instance
    rawDocuments = fileSystem.load_documents()
    print(f"RAW DOCUMENTS: {rawDocuments}")

    # Initialize an in-memory metadata DB
    metadata_db = InMemoryDocumentMetadataDB()

    # Parse the raw documents
    parsedDocuments = await MultiDocumentParser().parse_documents(
        rawDocuments,
        ParserConfig(
            metadata_db=metadata_db,
            access_control_policy_factory=AlwaysAllowDocumentAccessPolicyFactory(),
        ),
    )

    # Initialize a document transformer
    # TODO set parameters better
    documentTransformer = SeparatorTextChunker(
        SeparatorTextChunkerParams(
            metadata_db=metadata_db,
            text_chunk_config=TextChunkConfig(
                chunk_size_limit=500, chunk_overlap=100, size_fn=len
            ),
        )
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
    )

    openaiembcfg = OpenAIEmbeddingsConfig(api_key=config.openai_key)

    embeddings = OpenAIEmbeddings(openaiembcfg)
    pineconeVectorDB = await PineconeVectorDB.from_documents(
        transformedDocuments, pcvdbcfg, embeddings, metadata_db
    )

    # TODO: validate state of pineconeVectorDB
    print(f"{pineconeVectorDB=}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv))
