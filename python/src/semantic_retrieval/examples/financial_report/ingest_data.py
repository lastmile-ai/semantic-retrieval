import asyncio

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
)

from semantic_retrieval.data_store.vector_dbs.pinecone_vector_db import (
    PineconeVectorDB,
    PineconeVectorDBConfig,
)

from semantic_retrieval.transformation.embeddings.openai_embeddings import OpenAIEmbeddings


async def main():
    # Create a new FileSystem instance
    fileSystem = FileSystem("examples/example_data/financial_report")

    # Load documents using the FileSystem instance
    rawDocuments = fileSystem.load_documents()

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
    documentTransformer = SeparatorTextChunker()

    # Transform the parsed documents
    transformedDocuments = await documentTransformer.transform_documents(parsedDocuments)

    # Generate a new namespace using UUID
    namespace = "ns123"
    print(f"NAMESPACE: {namespace}")

    # Create a PineconeVectorDB instance and index the transformed documents
    pineconeVectorDB = PineconeVectorDB.from_documents(
        transformedDocuments,
        PineconeVectorDBConfig(
            embeddings=OpenAIEmbeddings(),
            metadata_db=metadata_db,
            index_name="test-financial-report-py",
            namespace=namespace,
        ),
    )

    # TODO: validate state of pineconeVectorDB
    print(f"{pineconeVectorDB=}")


if __name__ == "__main__":
    asyncio.run(main())
