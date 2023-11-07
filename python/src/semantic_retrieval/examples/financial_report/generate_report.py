import asyncio
from functools import partial
import json
import logging
import os
import re
import sys
from typing import List

import pandas as pd
from semantic_retrieval.access_control.access_function import AccessFunction
from semantic_retrieval.access_control.access_identity import AuthenticatedIdentity

from semantic_retrieval.common.core import LOGGER_FMT, file_contents
from semantic_retrieval.document.metadata.document_metadata import DocumentMetadata
from semantic_retrieval.document.metadata.document_metadata_db import DocumentMetadataDB

import semantic_retrieval.examples.financial_report.financial_report_generator as frg
from result import Err, Ok, Result
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
)
from semantic_retrieval.examples.financial_report.financial_report_generator import (
    FinancialReportGenerator,
)
from semantic_retrieval.examples.financial_report.lib import portfolio_df_to_dict
from semantic_retrieval.retrieval.csv_retriever import CSVRetriever
from semantic_retrieval.transformation.embeddings.openai_embeddings import (
    OpenAIEmbeddingsConfig,
)

import semantic_retrieval.examples.financial_report.financial_report_document_retriever as frdr

from semantic_retrieval.utils import callbacks as lib_callbacks

logger = logging.getLogger(__name__)
logging.basicConfig(format=LOGGER_FMT)


class FinancialReport(Record):
    pass


async def main(argv: List[str]):
    loggers = [logger, frg.logger, frdr.logger, lib_callbacks.logger]

    args = set_up_script(argv, loggers)
    config = get_config(args)
    logger.debug("CONFIG:\n")
    logger.debug(str(config))

    return await run_generate_report(config)


async def run_generate_report(config: Config):
    metadata_path = resolve_path(config.data_root, config.metadata_db_path)
    res_metadata_db = await InMemoryDocumentMetadataDB.from_json_file(metadata_path)

    callback_manager = lib_callbacks.CallbackManager(
        [
            lib_callbacks.to_json(
                "examples/example_data/financial_report/artifacts/callback_data.json"
            )
        ]
    )

    match res_metadata_db:
        case Err(msg):
            print(f"Error loading metadataDB: {msg}")
            return -1
        case Ok(metadata_db):
            openai_embedding_config = OpenAIEmbeddingsConfig(api_key=config.openai_key)

            pinecone_vectordb_config = PineconeVectorDBConfig(
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

            # TODO [P1]: This is where we would authenticate the viewer.
            viewer_identity = AuthenticatedIdentity(viewer_auth_id=config.viewer_role)
            portfolio_access_function: AccessFunction = validate_portfolio_access
            portfolio_retriever = CSVRetriever(
                file_path=resolve_path(config.data_root, portfolio_csv_path),
                viewer_identity=viewer_identity,
                user_access_function=portfolio_access_function,
                callback_manager=callback_manager,
            )

            res_portfolio: Result[
                pd.DataFrame, str
            ] = await portfolio_retriever.retrieve_data()

            match res_portfolio:
                case Err(msg):
                    print(f"Error loading portfolio: {msg}")
                case Ok(df_portfolio):
                    report = await _generate_report_for_portfolio(
                        df_portfolio,
                        pinecone_vectordb_config=pinecone_vectordb_config,
                        openai_embedding_config=openai_embedding_config,
                        metadata_db=metadata_db,
                        config=config,
                        viewer_identity=viewer_identity,
                        callback_manager=callback_manager,
                    )

                    # TODO [P1]: Save res to disk
                    print("Report:\n")
                    print(report.unwrap_or_else(lambda err: f"Error: {err}"))


async def _generate_report_for_portfolio(
    df_portfolio: pd.DataFrame,
    pinecone_vectordb_config: PineconeVectorDBConfig,
    openai_embedding_config: OpenAIEmbeddingsConfig,
    metadata_db: InMemoryDocumentMetadataDB,
    viewer_identity: AuthenticatedIdentity,
    config: Config,
    callback_manager: lib_callbacks.CallbackManager,
) -> Result[str, str]:
    portfolio = portfolio_df_to_dict(df_portfolio)
    logger.info("\nPortfolio:\n" + json.dumps(portfolio, indent=2))

    access_function_10k: AccessFunction = partial(
        validate_10k_access,
        metadata_db=metadata_db,
    )
    retriever = FinancialReportDocumentRetriever(
        vector_db_config=pinecone_vectordb_config,
        embeddings_config=openai_embedding_config,
        portfolio=portfolio,
        metadata_db=metadata_db,
        viewer_identity=viewer_identity,
        user_access_function=access_function_10k,
        callback_manager=callback_manager,
    )

    generator = FinancialReportGenerator(callback_manager=callback_manager)

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

    return Ok(res)


async def validate_portfolio_access(resource_auth_id: str, viewer_auth_id: str) -> bool:
    # In production, this function could do an IAM lookup, DB access, etc.
    # For this simulation, we read from a local JSON file.

    # In this case, the resource_auth_id is the csv path
    # and the viewer_auth_id is the advisor name.
    basename = os.path.basename(resource_auth_id)
    re_client_name = re.search(r"(.*)_portfolio.csv", basename)
    if not re_client_name:
        return False

    client_name = re_client_name.groups()[0]

    # User can look this up in real DB.
    path = "python/src/semantic_retrieval/examples/financial_report/access_control/iam_simulation_db.json"
    db_iam_simulation = json.loads(file_contents(path))["advisors"]

    return db_iam_simulation.get(client_name, None) == viewer_auth_id


async def validate_10k_access(
    resource_auth_id: str, viewer_auth_id: str, metadata_db: DocumentMetadataDB
) -> bool:
    def _validate_10k_access_with_metadata(
        resource_auth_id: str,
        viewer_auth_id: str,
        metadata: DocumentMetadata,
    ):
        uri = metadata.uri
        ticker_re = re.search(r".*_([A-Z\.]+)\..*", uri)
        if not ticker_re:
            return False
        ticker = str(ticker_re.groups()[0])
        # if ticker_re else ""
        # print(f"{out=}")

        logger.debug(
            f"validate_10k_access({resource_auth_id=}, {viewer_auth_id=}, {ticker=}"
        )
        path = "python/src/semantic_retrieval/examples/financial_report/access_control/iam_simulation_db.json"
        db_iam_simulation = json.loads(file_contents(path))["access_10ks"]
        return ticker in db_iam_simulation.get(viewer_auth_id, [])

    res_metadata = await metadata_db.get_metadata(resource_auth_id)

    match res_metadata:
        case Err(_msg):
            # TODO log
            return False
        case Ok(metadata):
            return _validate_10k_access_with_metadata(
                metadata=metadata,
                viewer_auth_id=viewer_auth_id,
                resource_auth_id=resource_auth_id,
            )


if __name__ == "__main__":
    res = asyncio.run(main(sys.argv))
    sys.exit(res)
