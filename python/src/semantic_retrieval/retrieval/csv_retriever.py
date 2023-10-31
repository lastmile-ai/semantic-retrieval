from typing import Generic

import pandas as pd

from semantic_retrieval.common.types import R, Record
from semantic_retrieval.retrieval.retriever import (
    BaseRetriever,
    BaseRetrieverQueryParams,
)


class CSVRetrieverQuery(Record):
    # For now, assume a single primary key column
    primary_key_column: str


class CSVRetriever(Generic[R], BaseRetriever[R, CSVRetrieverQuery]):
    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path

    async def retrieve_data(
        self, params: BaseRetrieverQueryParams[CSVRetrieverQuery]
    ) -> R:
        return (
            pd.read_csv(self.file_path)
            .set_index("Company")
            .fillna(0)
            .query("Shares > 0")["Shares"]
            .to_dict()
        )  # type: ignore [fixme]
