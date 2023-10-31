import argparse
import asyncio
import os
import sys
from typing import List
from dotenv import load_dotenv

from result import Err, Ok
from semantic_retrieval.common.types import Record
from semantic_retrieval.data_store.vector_dbs.pinecone_vector_db import (
    PineconeVectorDB,
    PineconeVectorDBConfig,
)
from semantic_retrieval.data_store.vector_dbs.vector_db import (
    VectorDBTextQuery,
)

from semantic_retrieval.document.metadata.in_memory_document_metadata_db import (
    InMemoryDocumentMetadataDB,
)

from semantic_retrieval.examples.financial_report.access_control.identities import (
    AdvisorIdentity,
)
from semantic_retrieval.examples.financial_report.config import Config, argparsify
from semantic_retrieval.retrieval.csv_retriever import CSVRetriever
from semantic_retrieval.retrieval.vector_dbs.vector_db_document_retriever import (
    VectorDBDocumentRetriever,
)

from semantic_retrieval.transformation.embeddings.openai_embeddings import (
    OpenAIEmbeddings,
    OpenAIEmbeddingsConfig,
)


from semantic_retrieval.access_control.access_passport import AccessPassport

from semantic_retrieval.utils.configs.configs import combine_dicts, remove_nones


class FinancialReport(Record):
    pass


def resolve_path(data_root: str, path: str) -> str:
    """
    data_root is relative to csv
    path is relative to data_root
    """

    return os.path.join(os.getcwd(), data_root, path)


async def main(argv: List[str]):
    load_dotenv()

    parser = argparsify(Config)
    args = parser.parse_args(argv[1:])

    config = get_config(args)

    return await run_generate_report(config)


def get_config(args: argparse.Namespace):
    # TODO combine stuff cleaner
    args_resolved = combine_dicts(
        [
            remove_nones(d)
            for d in [
                vars(args),
                dict(
                    openai_key=os.getenv("OPENAI_API_KEY"),
                    pinecone_key=os.getenv("PINECONE_API_KEY"),
                ),
            ]
        ]
    )
    # print(args_resolved)

    return Config(**args_resolved)


async def run_generate_report(config: Config):
    # Load the metadataDB persisted from ingest_data script
    metadata_path = resolve_path(config.data_root, config.metadata_db_path)
    res_metadata_db = await InMemoryDocumentMetadataDB.from_json_file(metadata_path)

    match res_metadata_db:
        case Err(msg):
            print(f"Error loading metadataDB: {msg}")
            return -1
        case Ok(metadata_db):
            print(f"{metadata_db.metadata=}")
            vdbcfg = PineconeVectorDBConfig(
                index_name=config.index_name,
                namespace=config.namespace,
            )
            openaiembcfg = OpenAIEmbeddingsConfig(api_key=config.openai_key)

            embeddings = OpenAIEmbeddings(openaiembcfg)
            vector_db = PineconeVectorDB(
                vdbcfg,
                embeddings=embeddings,
                metadata_db=metadata_db,
            )

            query_res = await vector_db.query(
                VectorDBTextQuery(
                    mode="text", metadataFilter={}, topK=10, text="cash flow"
                )
            )
            print(f"{query_res=}")

            _document_retriever = VectorDBDocumentRetriever(
                vector_db=vector_db,
                metadata_db=metadata_db,
            )

            _portfolio_retriever = CSVRetriever(
                resolve_path(config.data_root, config.portfolio_csv_path)
            )

            access_passport = AccessPassport()
            identity = AdvisorIdentity(client=config.client_name)
            access_passport.register(identity)

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
    res = asyncio.run(main(sys.argv))
    sys.exit(res)
