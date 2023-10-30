from enum import Enum
from typing import Optional

from semantic_retrieval.access_control.access_identity import AccessIdentity


class Role(Enum):
    USER = "user"
    ADMIN = "admin"


class FinancialReportIdentity(AccessIdentity):
    resource: str = "financial_data"
    role: Role
    client: Optional[str]


class AdvisorIdentity(FinancialReportIdentity):
    resource = "financial_data"
    role: Role = Role.USER
    client = None

    def __init__(self, client: str) -> None:
        self.client = client
        # TODO
        pass
