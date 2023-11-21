import json
from abc import abstractmethod
from typing import Any, List, Optional, Union

from semantic_retrieval.access_control.access_identity import AccessIdentity
from semantic_retrieval.common.types import Record
from semantic_retrieval.document.document import Document


# Access policy for a resource
class ResourceAccessPolicy(Record):
    policy: str
    resource: Optional[str] = None
    policyJSON: Optional[Any] = None

    @abstractmethod
    async def testDocumentReadPermission(
        self, document: Document, requestor: AccessIdentity
    ) -> bool:
        # Implement the testDocumentReadPermission logic
        pass

    @abstractmethod
    async def testPolicyPermission(
        self, requestor: AccessIdentity
    ) -> Union[List[str], bool]:
        pass


# An in-memory cache of the rest of test*Permission calls
class ResourceAccessPolicyCache:
    def __init__(self):
        self.cache = {}

    def get(
        self, policy: str, requestor: AccessIdentity
    ) -> Union[List[str], bool, None]:
        key = json.dumps(requestor) + policy
        return self.cache.get(key)

    def set(
        self,
        policy: str,
        requestor: AccessIdentity,
        permissions: Union[List[str], bool],
    ):
        key = json.dumps(requestor) + policy
        self.cache[key] = permissions
