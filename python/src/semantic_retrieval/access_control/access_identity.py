from pydantic.dataclasses import dataclass
from semantic_retrieval.common.base import Attributable


@dataclass
class AccessIdentity(Attributable):
    resource: str
