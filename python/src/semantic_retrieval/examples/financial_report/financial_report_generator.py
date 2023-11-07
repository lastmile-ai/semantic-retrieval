import json
import logging
from typing import List
from semantic_retrieval.common.core import LOGGER_FMT
from semantic_retrieval.examples.financial_report.financial_report_document_retriever import (
    FinancialReportData,
    FinancialReportDocumentRetriever,
    PortfolioData,
)

import openai

from semantic_retrieval.utils.callbacks import CallbackManager, Traceable


logger = logging.getLogger(__name__)
logging.basicConfig(format=LOGGER_FMT)


class FinancialReportGenerator(Traceable):
    def __init__(self, callback_manager: CallbackManager) -> None:
        self.callback_manager = callback_manager

    async def run(
        self,
        # access_passport,
        portfolio: PortfolioData,
        system_prompt: str,
        retrieval_query: str,
        structure_prompt: str,
        data_extraction_prompt: str,
        top_k: int,
        overfetch_factor: float,
        retriever: FinancialReportDocumentRetriever,
    ):
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

        retrieved_data_processed = process_retrieved_data(portfolio, retrieved_data)

        system_content = system_prompt

        structure = (
            f"STRUCTURE: {structure_prompt} containing the {data_extraction_prompt}.\n"
        )
        logger.info("Requested report:")
        logger.info(structure)
        user_content = structure + "CONTEXT:\n" + "\n * ".join(retrieved_data_processed)

        result = _generate(system_content, user_content)

        return result


def process_retrieved_data(
    portfolio: PortfolioData, retrieved_data: List[FinancialReportData]
) -> List[str]:
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

    return [
        f"Security: {company}\nDetails: {details}"
        for company, details in portfolio_with_details.items()
        if company in portfolio.keys()
    ]


def _generate(system_content: str, user_content: str) -> str:
    # TODO [P0.5]: implement with aiconfig
    system = {
        "role": "system",
        "content": system_content,
    }
    logger.debug("system content:\n")
    logger.debug(system_content)
    logger.debug("\n\nuser_content:\n")
    logger.debug(user_content)

    response = openai.ChatCompletion.create(  # type: ignore [fixme]
        model="gpt-4",
        messages=[
            system,
            {"role": "user", "content": user_content},
        ],
    )

    return response.choices[0]["message"]["content"]  # type: ignore [fixme]
