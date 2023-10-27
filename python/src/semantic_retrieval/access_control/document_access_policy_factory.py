from abc import ABC, abstractmethod
from typing import Awaitable, List
from semantic_retrieval.access_control.resource_access_policy import (
    ResourceAccessPolicy,
)

from semantic_retrieval.document.document import RawDocument


class DocumentAccessPolicyFactory(ABC):
    @abstractmethod
    def get_access_policies(
        self, raw_document: RawDocument
    ) -> Awaitable[List[ResourceAccessPolicy]]:
        pass
