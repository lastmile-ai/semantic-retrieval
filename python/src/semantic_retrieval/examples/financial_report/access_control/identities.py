from enum import Enum
from typing import Optional

from semantic_retrieval.access_control.access_identity import AccessIdentity


class Role(Enum):
    USER = "user"
    ADMIN = "admin"


class FinancialReportIdentity(AccessIdentity):
    resource: str = "financial_data"
    role: Role


class AdvisorIdentity(FinancialReportIdentity):
    role: Role = Role.USER
    client: Optional[str] = None
