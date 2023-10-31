
from abc import abstractmethod, ABC

from semantic_retrieval.document.document import IngestedDocument, RawDocument


class DocumentParser(ABC):
    @abstractmethod
    async def parse(self, document: RawDocument) -> IngestedDocument:
        pass
