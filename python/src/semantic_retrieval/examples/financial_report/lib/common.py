from typing import Dict, NewType
import pandas as pd
from semantic_retrieval.common.types import Record

PortfolioData = NewType("PortfolioData", Dict[str, int])


class FinancialReportData(Record):
    company: str
    details: str


def portfolio_df_to_dict(df: pd.DataFrame) -> PortfolioData:
    return PortfolioData(
        df.set_index("Company").astype(float).fillna(0).query("Shares > 0")["Shares"].to_dict()
    )
