from abc import ABC
from typing import Any, Optional


class Attributable(ABC):
    metadata: Optional[dict[Any, Any]]
    attributes: Optional[dict[Any, Any]]


class Identifiable(ABC):
    id: str
