import json
import logging
from typing import List
from semantic_retrieval.common.core import LOGGER_FMT
from semantic_retrieval.examples.financial_report.lib import financial_report_document_retriever
from semantic_retrieval.examples.financial_report.lib.common import (
    FinancialReportData,
    PortfolioData,
)
from semantic_retrieval.generator.retrieval_augmented_generation.generator import (
    generate,
    resolve_ai_config,
)

from semantic_retrieval.utils.callbacks import CallbackManager, Traceable


logger = logging.getLogger(__name__)
logging.basicConfig(format=LOGGER_FMT)


class FinancialReportGenerator(Traceable):
    def __init__(self, callback_manager: CallbackManager) -> None:
        self.callback_manager = callback_manager

    async def run(
        self,
        portfolio: PortfolioData,
        retrieval_query: str,
        top_k: int,
        overfetch_factor: float,
        retriever: financial_report_document_retriever.FinancialReportDocumentRetriever,
    ) -> str:
        res_retrieved_data = await retriever.retrieve_data(
            portfolio=portfolio,
            query=retrieval_query,
            top_k=top_k,
            overfetch_factor=overfetch_factor,
        )

        if res_retrieved_data.is_err():
            raise Exception(res_retrieved_data.err())

        retrieved_data = res_retrieved_data.unwrap()

        logger.debug("Raw retrieved data:")
        logger.debug(json.dumps([rd.model_dump() for rd in retrieved_data], indent=2))

        formatted_retrieved_data = process_retrieved_data(portfolio, retrieved_data)

        resolved = await resolve_ai_config(
            ai_config_path="python/src/semantic_retrieval/aiconfigs/py-completion-gen-aiconfig_aiconfig.json",
            params=dict(data=formatted_retrieved_data),
        )
        requested_report = resolved["messages"][1]["content"].split("\n")[0]
        logger.info(f"Requested report:\n{retrieval_query=}\n{requested_report}\n\n")

        result = await generate(
            ai_config_path="python/src/semantic_retrieval/aiconfigs/py-completion-gen-aiconfig_aiconfig.json",
            params=dict(data=formatted_retrieved_data),
        )

        return result


def process_retrieved_data(
    portfolio: PortfolioData, retrieved_data: List[FinancialReportData]
) -> str:
    portfolio_with_details = {}
    for fr_data in retrieved_data:
        company = fr_data.company
        if company not in portfolio_with_details:
            portfolio_with_details[company] = fr_data.details

    if portfolio_with_details.keys() != portfolio.keys():
        print(
            "WARNING: some companies are missing from the retrieved data. Try increasing overfetch_factor.\n"
            f"found={portfolio_with_details.keys() & portfolio.keys()}\n"
            f"missing={portfolio.keys() - portfolio_with_details.keys()}\n"
        )

    the_list = [
        f"Security: {company}\nDetails: {details}"
        for company, details in portfolio_with_details.items()
        if company in portfolio.keys()
    ]

    return "\n * ".join(the_list)