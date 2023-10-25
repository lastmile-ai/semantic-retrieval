from semantic_retrieval.ingestion.data_sources.data_source import DataSource
from semantic_retrieval.document.document import RawDocument


class FileSystem(DataSource):
    name: str = "FileSystem"
    path: str

    def __init__(self, path: str):
        self.path = path

    def load_documents(
        self, filters: any = None, limit: int = None
    ) -> list[RawDocument]:
        return []
