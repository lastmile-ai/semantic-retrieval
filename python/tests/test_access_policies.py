import pytest
from semantic_retrieval.access_control.access_identity import AccessIdentity
from semantic_retrieval.access_control.access_passport import AccessPassport
from semantic_retrieval.access_control.policies.always_allow_access_policy import (
    AlwaysAllowAccessPolicy,
)
from semantic_retrieval.access_control.resource_access_policy import (
    ResourceAccessPolicy,
)


class AlwaysDenyPolicy(ResourceAccessPolicy):
    async def testDocumentReadPermission(self, document, requestor):  # type: ignore [fixme]
        return False

    async def testPolicyPermission(self, requestor):  # type: ignore [fixme]
        return False


def test_access_passport():
    access_passport = AccessPassport()
    access_passport.register(AccessIdentity("test-resource"))

    assert access_passport.get_identity("test-resource").resource == "test-resource"


@pytest.mark.asyncio
async def test_access_policies():
    always_deny_policy = AlwaysDenyPolicy("always_deny")
    always_accept_policy = AlwaysAllowAccessPolicy()

    assert always_deny_policy.policy == "always_deny"
    assert await always_deny_policy.testDocumentReadPermission(None, None) == False  # type: ignore [fixme]
    assert await always_deny_policy.testPolicyPermission(None) == False  # type: ignore [fixme]

    assert await always_accept_policy.testDocumentReadPermission(None, None) == True  # type: ignore [fixme]
    assert await always_accept_policy.testPolicyPermission(None) == True  # type: ignore [fixme]
