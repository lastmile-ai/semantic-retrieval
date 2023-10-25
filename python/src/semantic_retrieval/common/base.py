
from abc import ABC


class Attributable(ABC):
    metadata: dict[any, any] | None
    attributes: dict[any, any] | None


class Identifiable(ABC):
    id: str
