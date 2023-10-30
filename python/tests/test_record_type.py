from pydantic import ValidationError
import pytest
from semantic_retrieval.common.types import Record


class TheRecord(Record):
    x: int


def test_immutable():
    r = TheRecord(x=1)
    with pytest.raises(ValidationError):
        r.x = 2


def test_strict():
    with pytest.raises(ValidationError):
        TheRecord(x="1")  # type: ignore [this typeerror is intentional]
