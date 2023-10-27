from abc import ABC, abstractmethod
import json
from typing import Optional, List, Union
from semantic_retrieval.common.json_types import JSONObject

from semantic_retrieval.document.document import Document
from semantic_retrieval.access_control.access_identity import AccessIdentity


# Access policy for a resource
class ResourceAccessPolicy(ABC):
    def __init__(
        self,
        policy: str,
        resource: Optional[str] = None,
        policyJSON: Optional[JSONObject] = None,
    ):
        self.policy = policy
        self.resource = resource
        self.policyJSON = policyJSON

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
        # Implement the testPolicyPermission logic
        # TODO
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
