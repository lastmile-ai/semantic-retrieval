# Don't rely on the generic type. Wrong annotation might be missed.
# Use `Any` to signal that uncertainty explicitly.
import json
import time
from typing import Any, Awaitable, Callable, Generator, Optional, TypeVar

import numpy.typing as npt
from pydantic import BaseModel, ConfigDict
from result import Result

# TODO [P1]: is this useful?
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

TR = TypeVar("TR", covariant=True)
E = TypeVar("E", covariant=True)


class Record(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True)

    def __repr__(self) -> str:
        return json.dumps(self.model_dump(), indent=2)

    def __str__(self) -> str:
        return self.__repr__()

    def to_dict(self):
        return self.model_dump()


class CallbackEvent(Record):
    name: str
    # Anything available at the time the event happens.
    # It is passed to the callback.
    data: Any
    ts_ns: int = time.time_ns()


class CallbackResult(Record):
    result: Any


# Callbacks will run on every event with the run_id (str).
# They may have I/O side effects (e.g. logging) and/or return a CallbackResult.
# Any CallbackResults returned will be stored in the CallbackManager.
# The user can then access these results.
Callback = Callable[[CallbackEvent, str], Awaitable[Optional[CallbackResult]]]

# This is for annotating @do_result annotated functions.
# It will be treated as compatible with Result[TR, E].
DoResult = Generator[Result[TR, E], TR, Result[TR, E]]
