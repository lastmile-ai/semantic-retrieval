from typing import List
from semantic_retrieval.examples.financial_report.financial_report_document_retriever import (
    FinancialReportData,
    FinancialReportDocumentRetriever,
    PortfolioData,
)

import openai


class FinancialReportGenerator:
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
        retrieved_data = await retriever.retrieve_data(
            portfolio=portfolio,
            query=retrieval_query,
            top_k=top_k,
            overfetch_factor=overfetch_factor,
        )

        # print(f"{len(retrieved_data)=}")
        # for rd in retrieved_data:
        #     print(rd.company)
        #     print(len(rd.details))

        retrieved_data_processed = process_retrieved_data(portfolio, retrieved_data)

        system_content = system_prompt

        structure = (
            f"STRUCTURE: {structure_prompt} containing the {data_extraction_prompt}.\n"
        )
        print("Requested report:")
        print(structure)
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

    # print("RETRIEVE", retrieved_data[:2])

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
    system = {
        "role": "system",
        "content": system_content,
    }
    # print("system content:\n")
    # print(system_content)
    # print("\n\nuser_content:\n")
    # print(user_content)

    response = openai.ChatCompletion.create(  # type: ignore [fixme]
        model="gpt-4",
        messages=[
            system,
            {"role": "user", "content": user_content},
        ],
    )

    return response.choices[0]["message"]["content"]  # type: ignore [fixme]
