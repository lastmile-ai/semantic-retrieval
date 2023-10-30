from typing import Generic

from semantic_retrieval.common.types import R, Record
from semantic_retrieval.retrieval.retriever import BaseRetrieverQueryParams


class CSVRetrieverQuery(Record):
    # For now, assume a single primary key column
    primary_key_column: str


class CSVRetriever(Generic[R]):
    def __init__(self, filePath: str):
        super().__init__()
        self.file_path = filePath

    async def retrieve_data(self, params: BaseRetrieverQueryParams[CSVRetrieverQuery]):
        # TODO
        pass
