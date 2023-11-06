from typing import Optional
import pytest
from semantic_retrieval.access_control.access_identity import AccessIdentity
from semantic_retrieval.access_control.access_passport import AccessPassport
from semantic_retrieval.access_control.policies.always_allow_access_policy import (
    AlwaysAllowAccessPolicy,
)
from semantic_retrieval.access_control.resource_access_policy import (
    ResourceAccessPolicy,
)
from semantic_retrieval.common.core import make_run_id
from semantic_retrieval.document.document import Document
from semantic_retrieval.document.metadata.in_memory_document_metadata_db import (
    InMemoryDocumentMetadataDB,
)
from semantic_retrieval.document_parsers.multi_document_parser import (
    ParserConfig,
)
from semantic_retrieval.ingestion.data_sources.fs.file_system import FileSystem

import semantic_retrieval.document_parsers.multi_document_parser as mdp
from semantic_retrieval.utils.callbacks import CallbackManager

metadata_db = InMemoryDocumentMetadataDB(callback_manager=CallbackManager.default())


class AlwaysDenyPolicy(ResourceAccessPolicy):
    policy: str = "always_deny"

    async def testDocumentReadPermission(
        self, document: Document, requestor: Optional[AccessIdentity] = None
    ):
        return False

    async def testPolicyPermission(self, requestor: AccessIdentity):
        return False


def test_access_passport():
    access_passport = AccessPassport()
    access_passport.register(AccessIdentity(resource="test-resource"))

    def get_resource(ai: AccessIdentity) -> str:
        return ai.resource

    assert access_passport.get_identity("test-resource").map_or(False, get_resource)


@pytest.mark.asyncio
async def test_access_policies():
    always_deny_policy = AlwaysDenyPolicy()
    always_accept_policy = AlwaysAllowAccessPolicy()

    cm = CallbackManager.default()
    run_id = make_run_id()
    # Get ingested documents to be able to test the policies - TODO [P1]: This should either be helper or mocked
    file_system = FileSystem(
        "examples/example_data/financial_report",
        callback_manager=cm,
    )
    raw_documents = await file_system.load_documents(run_id)

    ingested_documents = await mdp.parse_documents(
        raw_documents,
        parser_config=ParserConfig(
            metadata_db=metadata_db, access_control_policy_factory=None
        ),
        callback_manager=cm,
        run_id=run_id,
    )

    assert always_deny_policy.policy == "always_deny"
    assert (
        await always_deny_policy.testDocumentReadPermission(
            ingested_documents[0], AccessIdentity(resource="abc")
        )
        == False
    )
    assert (
        await always_deny_policy.testPolicyPermission(AccessIdentity(resource="abc"))
        == False
    )

    assert (
        await always_accept_policy.testDocumentReadPermission(
            ingested_documents[0], AccessIdentity(resource="abc")
        )
        == True
    )
    assert (
        await always_accept_policy.testPolicyPermission(AccessIdentity(resource="abc"))
        == True
    )
