from abc import ABC, abstractmethod
from semantic_retrieval.document.document import RawDocument


class DataSource(ABC):
    @abstractmethod
    def load_documents(self, filters: any, limit: int) -> list[RawDocument]:
        pass
