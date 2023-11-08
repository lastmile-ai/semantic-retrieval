import asyncio
from functools import partial
import json
import logging
import os
import sys
from typing import List

import pandas as pd
from semantic_retrieval.access_control.access_function import AccessFunction
from semantic_retrieval.access_control.access_identity import AuthenticatedIdentity

from semantic_retrieval.common.core import LOGGER_FMT, text_file_write

from semantic_retrieval.examples.financial_report.lib import financial_report_generator
from result import Result
from semantic_retrieval.data_store.vector_dbs.pinecone_vector_db import (
    PineconeVectorDBConfig,
)
from semantic_retrieval.document.metadata.in_memory_document_metadata_db import (
    InMemoryDocumentMetadataDB,
)
from semantic_retrieval.examples.financial_report.lib import config
from semantic_retrieval.examples.financial_report.lib.financial_report_document_retriever import (
    FinancialReportDocumentRetriever,
)
from semantic_retrieval.examples.financial_report.lib.financial_report_generator import (
    FinancialReportGenerator,
)
from semantic_retrieval.examples.financial_report.lib.common import portfolio_df_to_dict
from semantic_retrieval.examples.financial_report.access_control import access_functions
from semantic_retrieval.retrieval.csv_retriever import CSVRetriever
from semantic_retrieval.transformation.embeddings.openai_embeddings import (
    OpenAIEmbeddingsConfig,
)

from semantic_retrieval.examples.financial_report.lib import financial_report_document_retriever

from semantic_retrieval.utils import callbacks as lib_callbacks

logger = logging.getLogger(__name__)
logging.basicConfig(format=LOGGER_FMT)


async def main(argv: List[str]):
    loggers = [
        logger,
        financial_report_generator.logger,
        financial_report_document_retriever.logger,
        lib_callbacks.logger,
    ]

    args = config.set_up_script(argv, loggers)
    config_instance = config.get_config(args)
    logger.debug("CONFIG:\n")
    logger.debug(str(config))

    return await run_generate_report(config_instance)


async def run_generate_report(config_instance: config.Config):
    # Make callback manager that writes all events to local JSON file
    callback_manager = lib_callbacks.CallbackManager(
        [
            lib_callbacks.to_json(
                "examples/example_data/financial_report/artifacts/callback_data.json"
            )
        ]
    )

    # Assume we have authenticated the viewer. In production,
    # we would get a true `AuthenticatedIdentity`.
    viewer_identity = AuthenticatedIdentity(viewer_auth_id=config_instance.viewer_role)

    # Set access policies: each data store has an access function.
    portfolio_access_function: AccessFunction = access_functions.validate_portfolio_access

    # Load in-memory metadata DB from local JSON.
    # This is used in conjunction with the document vector store.
    metadata_db_path = config.get_metadata_db_path(config_instance)
    res_metadata_db = await InMemoryDocumentMetadataDB.from_json_file(metadata_db_path)

    if res_metadata_db.is_err():
        print(f"Error loading metadataDB: {res_metadata_db}")
        return -1

    metadata_db = res_metadata_db.unwrap()

    access_function_10k: AccessFunction = partial(
        access_functions.validate_10k_access,
        metadata_db=metadata_db,
    )

    # Set the configs for embeddings and vector DB lookup
    openai_embedding_config = OpenAIEmbeddingsConfig(api_key=config_instance.openai_key)

    pinecone_vectordb_config = PineconeVectorDBConfig(
        index_name=config_instance.index_name,
        namespace=config_instance.namespace,
        api_key=config_instance.pinecone_key,
        environment=config_instance.pinecone_environment,
    )

    # Retrieve portfolio for client using CSVRetriever.
    logger.info(f"Client name: {config_instance.client_name}")
    portfolio_csv_name = f"{config_instance.client_name}_portfolio.csv"
    portfolio_csv_path = os.path.join(config_instance.portfolio_csv_dir, portfolio_csv_name)

    portfolio_retriever = CSVRetriever(
        file_path=config.resolve_path(config_instance.data_root, portfolio_csv_path),
        viewer_identity=viewer_identity,
        user_access_function=portfolio_access_function,
        callback_manager=callback_manager,
    )

    res_portfolio: Result[pd.DataFrame, str] = await portfolio_retriever.retrieve_data()
    if res_portfolio.is_err():
        print(f"Error loading portfolio: {res_portfolio}")
        return -1

    df_portfolio = res_portfolio.unwrap()
    dict_portfolio = portfolio_df_to_dict(df_portfolio)
    logger.info("\nPortfolio:\n" + json.dumps(dict_portfolio, indent=2))

    # Using the objects and data set up above, construct the
    # vector store retriever for chunks of 10ks related to the portfolio.
    retriever = FinancialReportDocumentRetriever(
        vector_db_config=pinecone_vectordb_config,
        embeddings_config=openai_embedding_config,
        portfolio=dict_portfolio,
        metadata_db=metadata_db,
        viewer_identity=viewer_identity,
        user_access_function=access_function_10k,
        callback_manager=callback_manager,
    )

    # Construct a completion generator. This is configured with
    # the AIConfig file py-completion-gen-aiconfig_aiconfig.json
    generator = FinancialReportGenerator(callback_manager=callback_manager)

    # Get the (string-valued) report from the generator.
    # This synthesizes the portfolio and uses the 10k retriever
    # to retrieve chunks of 10ks related to the portfolio.
    # Then it passes the relevant data to the generator.
    logger.info("Generating report...")
    report = await generator.run(
        dict_portfolio,
        top_k=config_instance.top_k,
        overfetch_factor=config_instance.overfetch_factor,
        retriever=retriever,
        ai_config_path=config_instance.ai_config_path,
        variant_name=config_instance.variant_name,
    )

    # Save for evaluation.
    res_write = text_file_write(config_instance.sample_output_path, report)
    print(f"\n\nReport written to {config_instance.sample_output_path}, {res_write=}")


if __name__ == "__main__":
    res = asyncio.run(main(sys.argv))
    sys.exit(res)
