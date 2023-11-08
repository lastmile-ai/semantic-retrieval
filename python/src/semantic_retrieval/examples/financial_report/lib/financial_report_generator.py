import json
import logging
from typing import Any, Dict, List
from semantic_retrieval.common.core import LOGGER_FMT
from semantic_retrieval.examples.financial_report.lib import financial_report_document_retriever
from semantic_retrieval.examples.financial_report.lib.common import (
    FinancialReportData,
    PortfolioData,
)
from semantic_retrieval.generator.retrieval_augmented_generation.generator import (
    ai_config_metadata_lookup,
    generate,
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
        top_k: int,
        overfetch_factor: float,
        retriever: financial_report_document_retriever.FinancialReportDocumentRetriever,
        ai_config_path: str,
        variant_name: str,
    ) -> str:
        params_variant = await get_ai_config_params_for_variant(ai_config_path, variant_name)
        requested_report = json.dumps(params_variant, indent=2)
        logger.info(f"Requested report:\n{requested_report}\n\n")

        retrieval_query = params_variant["retrieval_query"]

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

        ai_config_params = dict(data=formatted_retrieved_data, **params_variant)
        result = await generate(
            ai_config_path=ai_config_path,
            params=ai_config_params,
        )

        return result


async def get_ai_config_params_for_variant(
    ai_config_path: str, variant_name: str
) -> Dict[str, Any]:
    variants = ai_config_metadata_lookup(ai_config_path=ai_config_path, key="report_variants")

    try:
        return variants[variant_name]  # type: ignore
    except KeyError:
        raise Exception(
            f"Unknown variant name: {variant_name}. Available variants: {variants.keys()}"
        )


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
