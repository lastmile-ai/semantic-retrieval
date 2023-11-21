from typing import List, Optional, Union

from semantic_retrieval.access_control.access_identity import AccessIdentity

# from semantic_retrieval.access_control.policies.resource_access_policy import ResourceAccessPolicy
from semantic_retrieval.access_control.resource_access_policy import (
    ResourceAccessPolicy,
)
from semantic_retrieval.document.document import Document


class AlwaysAllowAccessPolicy(ResourceAccessPolicy):
    policy: str = "always_allow"

    async def testDocumentReadPermission(
        self, document: Document, requestor: Optional[AccessIdentity] = None
    ) -> bool:
        return True

    async def testPolicyPermission(
        self, requestor: AccessIdentity
    ) -> Union[List[str], bool]:
        return True
