import asyncio
import uuid

from semantic_retrieval.ingestion.data_sources.fs.file_system import FileSystem
from semantic_retrieval.document.metadata.in_memory_document_metadata_db import (
    InMemoryDocumentMetadataDB,
)

from semantic_retrieval.data_store.vector_dbs.pinecone_vector_db import PineconeVectorDB

from semantic_retrieval.transformation.embeddings.openai_embeddings import OpenAIEmbeddings


from semantic_retrieval.retrieval.document_retriever import DocumentRetriever

from semantic_retrieval.generator.retrieval_augmented_generation.vector_db_rag_completion_generator import (
    VectorDBRAGCompletionGenerator,
)

from semantic_retrieval.access_control.access_passport import AccessPassport

from semantic_retrieval.access_control.access_identity import AccessIdentity

from semantic_retrieval.prompts.prompt_template import PromptTemplate


class OpenAIChatModel:
    # TODO:
    pass

    async def run(self, args):
        return "TODO"


class FinancialReportDocumentRetriever(DocumentRetriever):
    def __init__(self, vector_db, metadata_db):
        self.vector_db = vector_db
        self.metadata_db = metadata_db


class FinancialReportGenerator(VectorDBRAGCompletionGenerator):
    def __init__(self, chat_model):
        self.chat_model = chat_model
        super().__init__(chat_model)


async def main():
    # TODO: This needs to be a metadata DB implementation that persists/loads from disk
    metadataDB = InMemoryDocumentMetadataDB()

    vector_db = PineconeVectorDB(
        {
            "indexName": "test-financial-report-py",
            "namespace": "GET NAMESPACE FROM ingest_data RUN",
            "embeddings": OpenAIEmbeddings(),
            "metadataDB": metadataDB,
        }
    )

    retriever = FinancialReportDocumentRetriever(vector_db=vector_db, metadata_db=metadataDB)
    generator = FinancialReportGenerator(OpenAIChatModel())

    accessPassport = AccessPassport()

    access_identity = AccessIdentity("id")
    accessPassport.register(access_identity)

    data = "TODO"

    prompt = PromptTemplate(
        f"Use the following data to construct a financial report matching the following format ... {data}"
    )

    res = await generator.run(
        {
            "accessPassport": accessPassport,
            "prompt": prompt,
            "retriever": retriever,
        }
    )

    print("Report:\n")
    print(res)


if __name__ == "__main__":
    asyncio.run(main())
