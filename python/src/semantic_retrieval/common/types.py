# Don't rely on the generic type. Wrong annotation might be missed.
# Use `Any` to signal that uncertainty explicitly.
from typing import Any, TypeVar

import numpy.typing as npt
from pydantic import BaseModel, ConfigDict


# TODO: is this useful?
NPA = npt.NDArray[Any]

ArrayLike = npt.ArrayLike


# Canonical typevar for generator params
T = TypeVar("T")

# Canonical typevar for generator return type
R = TypeVar("R")

# Canonical typevar for retriever query type
Q = TypeVar("Q")

# Canonical typevar for params
P = TypeVar("P")


class Record(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True)
