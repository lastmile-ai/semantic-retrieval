import argparse
import asyncio
import os
import sys
from typing import List
from dotenv import load_dotenv

from result import Err, Ok
from semantic_retrieval.common.types import Record
from semantic_retrieval.data_store.vector_dbs.pinecone_vector_db import (
    PineconeVectorDBConfig,
)

from semantic_retrieval.document.metadata.in_memory_document_metadata_db import (
    InMemoryDocumentMetadataDB,
)

from semantic_retrieval.examples.financial_report.config import Config, argparsify
from semantic_retrieval.examples.financial_report.financial_report_document_retriever import (
    FinancialReportDocumentRetriever,
    PortfolioData,
)
from semantic_retrieval.examples.financial_report.financial_report_generator import (
    FinancialReportGenerator,
)
from semantic_retrieval.retrieval.csv_retriever import CSVRetriever

from semantic_retrieval.transformation.embeddings.openai_embeddings import (
    OpenAIEmbeddingsConfig,
)


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
            openaiembcfg = OpenAIEmbeddingsConfig(api_key=config.openai_key)

            pcvdbcfg = PineconeVectorDBConfig(
                index_name=config.index_name,
                namespace=config.namespace,
                api_key=config.pinecone_key,
                environment=config.pinecone_environment,
            )

            portfolio_csv_name = f"{config.client_name}_portfolio.csv"
            portfolio_csv_path = os.path.join(
                config.portfolio_csv_dir, portfolio_csv_name
            )
            portfolio_retriever = CSVRetriever(
                resolve_path(config.data_root, portfolio_csv_path)
            )

            portfolio: PortfolioData = await portfolio_retriever.retrieve_data(None)  # type: ignore [fixme]
            print(f"{portfolio=}")

            # access_passport = AccessPassport()
            # identity = AdvisorIdentity(client=config.client_name)
            # access_passport.register(identity)

            retriever = FinancialReportDocumentRetriever(
                #   access_passport,
                vector_db_config=pcvdbcfg,
                embeddings_config=openaiembcfg,
                portfolio=portfolio,  # type: ignore [fixme]
                metadata_db=metadata_db,
            )

            generator = FinancialReportGenerator()

            retrieval_query = config.retrieval_query

            system_prompt = (
                "INSTRUCTIONS:\n"
                "You are a helpful assistant. "
                "Rearrange the context to answer the question. "
                "Output your response following the requested structure. "
                "Do not include Any words that do not appear in the context. "
            )
            res = await generator.run(
                portfolio,
                system_prompt,
                retrieval_query,
                structure_prompt="Numbered List",
                data_extraction_prompt=config.data_extraction_prompt,
                top_k=config.top_k,
                overfetch_factor=config.overfetch_factor,
                retriever=retriever,
            )

            # TODO: Save res to disk and/or print
            print("Report:\n")
            print(res)


if __name__ == "__main__":
    res = asyncio.run(main(sys.argv))
    sys.exit(res)
