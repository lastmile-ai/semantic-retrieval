import asyncio
from semantic_retrieval.common.types import Record
from semantic_retrieval.data_store.vector_dbs.pinecone_vector_db import (
    PineconeVectorDB,
    PineconeVectorDBConfig,
)

from semantic_retrieval.document.metadata.in_memory_document_metadata_db import (
    InMemoryDocumentMetadataDB,
)

from semantic_retrieval.examples.financial_report.access_control.identities import AdvisorIdentity
from semantic_retrieval.retrieval.csv_retriever import CSVRetriever
from semantic_retrieval.retrieval.vector_dbs.vector_db_document_retriever import (
    VectorDBDocumentRetriever,
)
from semantic_retrieval.retrieval.vector_dbs.vector_db_retriever import VectorDBRetrieverParams

from semantic_retrieval.transformation.embeddings.openai_embeddings import (
    OpenAIEmbeddings,
)


from semantic_retrieval.access_control.access_passport import AccessPassport


class FinancialReport(Record):
    pass


async def main():
    # TODO
    pass

    # Load the metadataDB persisted from ingest_data script
    metadata_db = await InMemoryDocumentMetadataDB.from_json_file(
        "examples/python/financial_report/metadata_db.json"
    )

    vector_db = PineconeVectorDB(
        PineconeVectorDBConfig(
            index_name="test-financial-report-py",
            # TODO: Make this dynamic via script param
            namespace="the_namespace",
            embeddings=OpenAIEmbeddings(),
            metadata_db=metadata_db,
        )
    )

    _document_retriever = VectorDBDocumentRetriever(
        VectorDBRetrieverParams(
            vector_db=vector_db,
            metadata_db=metadata_db,
        )
    )

    # TODO: Make this dynamic via script param
    _portfolio_retriever = CSVRetriever(
        "examples/example_data/financial_report/portfolios/client_a_portfolio.csv"
    )

    access_passport = AccessPassport()
    identity = AdvisorIdentity("client_a")  # TODO: Make this dynamic via script param
    access_passport.register(identity)

    # TODO
    pass

    # retriever = FinancialReportDocumentRetriever({
    #   access_passport,
    #   document_retriever,
    #   portfolio_retriever,
    #   metadata_db,
    # })

    # generator = FinancialReportGenerator({
    #   model: OpenAIChatModel(),
    #   retriever,
    # })

    # prompt = PromptTemplate("Use the following data to construct a financial report matching the following format ... {data}")

    #   res = await generator.run({
    #     access_passport, # not necessary in this case, but include for example
    #     prompt,
    #     retriever,
    #   })

    # TODO: Save res to disk and/or print
    print("Report:\n")
    # print(res)


if __name__ == "__main__":
    asyncio.run(main())
