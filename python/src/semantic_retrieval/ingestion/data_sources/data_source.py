from abc import ABC, abstractmethod
from semantic_retrieval.document.document import RawDocument


from typing import Any


class DataSource(ABC):
    @abstractmethod
    async def load_documents(self, filters: Any, limit: int) -> list[RawDocument]:
        pass
