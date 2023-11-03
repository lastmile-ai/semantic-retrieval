from abc import abstractmethod, ABC
from result import Result

from semantic_retrieval.document.document import IngestedDocument, RawDocument


class DocumentParser(ABC):
    @abstractmethod
    async def parse(self, document: RawDocument) -> Result[IngestedDocument, str]:
        pass
