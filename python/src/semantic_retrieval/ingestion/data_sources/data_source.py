from abc import ABC, abstractmethod
from typing import Any

from semantic_retrieval.document.document import RawDocument


class DataSource(ABC):
    @abstractmethod
    async def load_documents(
        self, filters: Any, limit: int
    ) -> list[RawDocument]:
        pass
