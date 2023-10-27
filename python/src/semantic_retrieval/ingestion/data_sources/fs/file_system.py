from typing import Any, Optional
from semantic_retrieval.ingestion.data_sources.data_source import DataSource
from semantic_retrieval.document.document import RawDocument


class FileSystem(DataSource):
    name: str = "FileSystem"
    path: str

    def __init__(self, path: str):
        self.path = path

    def load_documents(
        self, filters: Optional[Any] = None, limit: Optional[int] = None
    ) -> list[RawDocument]:
        return []
