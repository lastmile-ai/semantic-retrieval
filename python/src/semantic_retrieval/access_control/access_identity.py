from semantic_retrieval.common.base import Attributable
from semantic_retrieval.common.types import Record


class AccessIdentity(Attributable):
    resource: str


class AuthenticatedIdentity(Record):
    # TODO [P1]: make this actually do authentication.
    viewer_auth_id: str
