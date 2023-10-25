
from abc import ABC
from typing import Optional


class Attributable(ABC):
    metadata: Optional[dict[any, any]]
    attributes: Optional[dict[any, any]]


class Identifiable(ABC):
    id: str
