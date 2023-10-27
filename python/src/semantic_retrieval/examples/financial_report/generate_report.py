import asyncio
from dataclasses import dataclass
from typing import Optional
from semantic_retrieval.document.metadata.document_metadata_db import DocumentMetadataDB

from semantic_retrieval.document.metadata.in_memory_document_metadata_db import (
    InMemoryDocumentMetadataDB,
)

from semantic_retrieval.data_store.vector_dbs.pinecone_vector_db import (
    PineconeVectorDB,
    PineconeVectorDBConfig,
)
from semantic_retrieval.generator.completion_generator import LLMCompletionGeneratorParams
from semantic_retrieval.generator.completion_models.completion_model import (
    CompletionModel,
)
from semantic_retrieval.prompts.prompt import IPrompt
from semantic_retrieval.retrieval.retriever import BaseRetriever

from semantic_retrieval.transformation.embeddings.openai_embeddings import (
    OpenAIEmbeddings,
)


from semantic_retrieval.retrieval.document_retriever import DocumentRetriever

from semantic_retrieval.generator.retrieval_augmented_generation.vector_db_rag_completion_generator import (
    VectorDBRAGCompletionGenerator,
)

from semantic_retrieval.access_control.access_passport import AccessPassport

from semantic_retrieval.access_control.access_identity import AccessIdentity

from semantic_retrieval.prompts.prompt_template import PromptTemplate
from semantic_retrieval.utils.callbacks import CallbackManager


@dataclass
class FinancialReportGeneratorParams(LLMCompletionGeneratorParams):
    access_passport: AccessPassport
    retriever: BaseRetriever
    prompt: IPrompt


@dataclass
class FinancialReport:
    pass


class OpenAIChatModel(CompletionModel):
    # TODO
    pass


class FinancialReportDocumentRetriever(DocumentRetriever):
    def __init__(self, vector_db: PineconeVectorDB, metadata_db: DocumentMetadataDB):
        super().__init__(metadata_db)

    # TODO:
    pass


class FinancialReportGenerator(VectorDBRAGCompletionGenerator):
    def __init__(self, model: CompletionModel, callback_manager: Optional[CallbackManager] = None):
        super().__init__(model, callback_manager)
        # TODO
        pass

    async def run(self, params: FinancialReportGeneratorParams) -> FinancialReport:
        # TODO
        return FinancialReport()


async def main():
    # TODO: This needs to be a metadata DB implementation that persists/loads from disk
    metadata_db = InMemoryDocumentMetadataDB()

    vector_db = PineconeVectorDB(
        PineconeVectorDBConfig(
            embeddings=OpenAIEmbeddings(),
            metadata_db=metadata_db,
            index_name="test-financial-report-py",
            namespace="ns123",
        )
    )

    retriever = FinancialReportDocumentRetriever(vector_db=vector_db, metadata_db=metadata_db)
    generator = FinancialReportGenerator(OpenAIChatModel())

    access_passport = AccessPassport()

    access_identity = AccessIdentity("id")
    access_passport.register(access_identity)

    data = "TODO"

    prompt = PromptTemplate(
        f"Use the following data to construct a financial report matching the following format ... {data}"
    )

    res = await generator.run(
        FinancialReportGeneratorParams(
            access_passport=access_passport, prompt=prompt, retriever=retriever
        )
    )

    print("Report:\n")
    print(res)


if __name__ == "__main__":
    asyncio.run(main())
