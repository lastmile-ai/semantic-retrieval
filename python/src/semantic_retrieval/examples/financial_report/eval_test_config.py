from semantic_retrieval.examples.financial_report.lib import (
    test_1_2_data_muncher,
    test_3_4_data_muncher,
)


TEST_CASES = [
    (name, test_1_2_data_muncher, ip, op)
    for name, ip, op in [
        (
            "net_income_vs_retrieved",
            "artifacts/raw_retrieved_chunks_10k_net_income.json",
            "artifacts/portfolio_10k_net_income_report.txt",
        ),
        (
            "covid_vs_retrieved",
            "artifacts/raw_retrieved_chunks_10k_covid.json",
            "artifacts/portfolio_10k_covid_report.txt",
        ),
    ]
] + [
    (name, test_3_4_data_muncher, ip, op)
    for name, ip, op in [
        (
            "net_income_e2e",
            "portfolios/sarmad_portfolio.csv",
            "artifacts/portfolio_10k_net_income_report.txt",
        ),
        (
            "covid_e2e",
            "portfolios/sarmad_portfolio.csv",
            "artifacts/portfolio_10k_covid_report.txt",
        ),
    ]
]
