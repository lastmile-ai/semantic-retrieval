from enum import Enum
from typing import Optional

from semantic_retrieval.common.types import Record


class Role(Enum):
    ADVISOR = "advisor"
    CLIENT = "client"



class AdvisorIdentity(Record):
    role: Role = Role.ADVISOR
    client: Optional[str] = None

class ClientIdentity(Record):
    role: Role = Role.CLIENT
    name: str

FinancialReportIdentity = AdvisorIdentity | ClientIdentity