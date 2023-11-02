import asyncio
import json
import logging
import os
import sys
from typing import List

from semantic_retrieval.common.core import LOGGER_FMT
from semantic_retrieval.examples.financial_report.access_control.identities import AdvisorIdentity

import semantic_retrieval.examples.financial_report.financial_report_generator as frg
from result import Err, Ok
from semantic_retrieval.common.types import Record
from semantic_retrieval.data_store.vector_dbs.pinecone_vector_db import (
    PineconeVectorDBConfig,
)
from semantic_retrieval.document.metadata.in_memory_document_metadata_db import (
    InMemoryDocumentMetadataDB,
)
from semantic_retrieval.examples.financial_report.config import (
    Config,
    get_config,
    resolve_path,
    set_up_script,
)
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

import semantic_retrieval.examples.financial_report.financial_report_document_retriever as frdr

logger = logging.getLogger(__name__)
logging.basicConfig(format=LOGGER_FMT)


class FinancialReport(Record):
    pass


async def main(argv: List[str]):
    loggers = [logger, frg.logger, frdr.logger]

    args = set_up_script(argv, loggers)
    config = get_config(args)
    logger.debug("CONFIG:\n")
    logger.debug(str(config))

    return await run_generate_report(config)


async def run_generate_report(config: Config):
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

            logger.info(f"Client name: {config.client_name}")
            portfolio_csv_name = f"{config.client_name}_portfolio.csv"
            portfolio_csv_path = os.path.join(
                config.portfolio_csv_dir, portfolio_csv_name
            )
            portfolio_retriever = CSVRetriever(
                resolve_path(config.data_root, portfolio_csv_path)
            )

            portfolio: PortfolioData = await portfolio_retriever.retrieve_data(None)  # type: ignore [fixme]
            logger.info("\nPortfolio:\n" + json.dumps(portfolio, indent=2))

            viewer_identity = AdvisorIdentity(client="client_a")



            retriever = FinancialReportDocumentRetriever(
                viewer_identity,
                config.client_name,
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
                structure_prompt=config.structure_prompt,
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
