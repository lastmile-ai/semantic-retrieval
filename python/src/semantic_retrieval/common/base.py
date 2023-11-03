from abc import ABC
from typing import Any, Dict, Optional

from semantic_retrieval.common.types import Record


class Attributable(ABC, Record):
    metadata: Optional[Dict[Any, Any]] = None
    attributes: Optional[Dict[Any, Any]] = None


class Identifiable(ABC):
    id: str
