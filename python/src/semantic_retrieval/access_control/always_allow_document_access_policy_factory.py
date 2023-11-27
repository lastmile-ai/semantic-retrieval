from typing import List

from semantic_retrieval.access_control.document_access_policy_factory import (
    DocumentAccessPolicyFactory,
)
from semantic_retrieval.access_control.policies.always_allow_access_policy import (
    AlwaysAllowAccessPolicy,
    ResourceAccessPolicy,
)
from semantic_retrieval.document.document import RawDocument


class AlwaysAllowDocumentAccessPolicyFactory(DocumentAccessPolicyFactory):
    def __init__(self):
        pass

    async def get_access_policies(
        self, raw_document: RawDocument
    ) -> List[ResourceAccessPolicy]:
        return [AlwaysAllowAccessPolicy()]
