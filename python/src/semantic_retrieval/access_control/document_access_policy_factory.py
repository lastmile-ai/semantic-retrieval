from abc import ABC, abstractmethod
from typing import List

from semantic_retrieval.access_control.resource_access_policy import (
    ResourceAccessPolicy,
)
from semantic_retrieval.document.document import RawDocument


class DocumentAccessPolicyFactory(ABC):
    @abstractmethod
    async def get_access_policies(
        self, raw_document: RawDocument
    ) -> List[ResourceAccessPolicy]:
        pass
