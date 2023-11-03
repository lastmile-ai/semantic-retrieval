from typing import Any, Awaitable, Callable, Final, Optional, Sequence

from semantic_retrieval.common.types import Record


class Traceable:
    """
    Interface for classes that support callbacks.

    TODO: figure out a way to type-enforce this
    By extending Traceable, a class affirms that it
    * accepts and stores a CallbackManager on init,
    * calls `run_callbacks()` on the CallbackManager
      with the appropriate event arguments at the appropriate
      points in its implementation.
    """

    pass


class CallbackEvent(Record):
    # Anything available at the time the event happens.
    # It is passed to the callback.
    data: Any
    # Globally unique identifier for the (e2e) run.
    # Callbacks should include this in any logs written,
    # as it is necessary to stitch together separate
    # records for analysis.
    run_id: str


class CallbackResult(Record):
    result: Any


# Callbacks will run on every event.
# They may have I/O side effects (e.g. logging) and/or return a CallbackResult.
# Any CallbackResults returned will be stored in the CallbackManager.
# The user can then access these results.
Callback = Callable[[CallbackEvent], Awaitable[Optional[CallbackResult]]]


class CallbackManager:
    def __init__(self, callbacks: Sequence[Callback]) -> None:
        self.callbacks: Final[Sequence[Callback]] = callbacks
        self.results = []

    # TODO: statically type each event?
    # TODO: [optimization] index callbacks by event type?
    async def run_callbacks(self, event: CallbackEvent) -> Optional[CallbackResult]:
        # TODO
        # TODO: [optimization] do this storage more efficiently
        for callback in self.callbacks:
            result = await callback(event)
            if result is not None:
                self.results.append(result)
