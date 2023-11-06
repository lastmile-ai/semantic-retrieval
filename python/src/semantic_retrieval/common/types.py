# Don't rely on the generic type. Wrong annotation might be missed.
# Use `Any` to signal that uncertainty explicitly.
import time
from typing import Any, Awaitable, Callable, Optional, TypeVar

import numpy.typing as npt
from pydantic import BaseModel, ConfigDict


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


class Record(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True)


class CallbackEvent(Record):
    name: str
    # Anything available at the time the event happens.
    # It is passed to the callback.
    data: Any
    # Globally unique identifier for the (e2e) run.
    # Callbacks should include this in any logs written,
    # as it is necessary to stitch together separate
    # records for analysis.
    run_id: Optional[str]
    ts_ns: int = time.time_ns()


class CallbackResult(Record):
    result: Any


# Callbacks will run on every event.
# They may have I/O side effects (e.g. logging) and/or return a CallbackResult.
# Any CallbackResults returned will be stored in the CallbackManager.
# The user can then access these results.
Callback = Callable[[CallbackEvent], Awaitable[Optional[CallbackResult]]]
