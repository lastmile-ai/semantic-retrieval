import asyncio
import dataclasses
import json
import logging
from typing import (
    Any,
    Coroutine,
    Final,
    Optional,
    Sequence,
    TextIO,
    TypeVar,
    Union,
)
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel
from result import Err, Ok, Result
from semantic_retrieval.common.core import LOGGER_FMT
from semantic_retrieval.common.types import (
    Callback,
    CallbackEvent,
    CallbackResult,
)

logger = logging.getLogger(__name__)
logging.basicConfig(format=LOGGER_FMT)

T = TypeVar("T")


class Traceable:
    """
    Interface for classes that support callbacks.

    TODO [P1]: figure out a way to type-enforce this
    By extending Traceable, a class affirms that it
    * accepts and stores a CallbackManager on init,
    * calls `run_callbacks()` on the CallbackManager
      with the appropriate event arguments at the appropriate
      points in its implementation.
    """

    pass


async def run_thunk_safe(
    thunk: Coroutine[Any, Any, T], timeout: int
) -> Result[T, str]:
    try:
        task = asyncio.create_task(thunk)
        res = await asyncio.wait_for(task, timeout=timeout)
        return Ok(res)
    except BaseException as e:  # type: ignore
        # TODO [P1] log
        return Err(str(e))


class CallbackManager:
    def __init__(
        self, callbacks: Sequence[Callback], run_id: Optional[str] = None
    ) -> None:
        self.callbacks: Final[Sequence[Callback]] = callbacks
        self.reset_run_state(run_id=run_id)

    # TODO [P1]: statically type each event?
    # TODO [P1]: [optimization] index callbacks by event type?
    async def run_callbacks(self, event: CallbackEvent) -> None:
        for callback in self.callbacks:
            logger.debug(f"RUNINNG CALLBACK, {callback=} on {event.name=}")

            async def _thunk():
                return await callback(event, self.run_id)

            # TODO [P1]: unhardcode timeout
            # TODO [P1]: [optimization] do this storage more efficiently
            result = await run_thunk_safe(_thunk(), timeout=1)
            self.results.append(result)

    def reset_run_state(self, run_id: Optional[str] = None) -> None:
        self.results = []
        self.run_id = run_id or make_run_id()

    @classmethod
    def default(cls) -> "CallbackManager":
        return CallbackManager([to_json("/var/logs/callbacks.json")])


def safe_serialize_json(obj: Any, **kwargs: Any):
    def default(o: Any):
        return f"<<non-serializable: {type(o).__qualname__}>>"

    return json.dumps(obj, default=default, **kwargs)


def _maybe_redacted(key: str, value: Any) -> Any:
    if isinstance(value, str) and any(
        k in key.lower() for k in ["api_key", "apikey", "token", "secret"]
    ):
        return "<<redacted>>"
    else:
        return value


def safe_serialize_arbitrary_for_logging(
    data: Any, max_elements: Optional[int] = None, indent: str = ""
) -> str:
    max_elements = max_elements or 20
    if isinstance(data, pd.DataFrame):
        return str(data.head(2))
    match data:
        case Ok(ok):
            return safe_serialize_arbitrary_for_logging(
                {"Ok": ok}, max_elements, indent + "  "
            )
        case Err(err):
            return safe_serialize_arbitrary_for_logging(
                {"Err": err}, max_elements, indent + "  "
            )
        case _:
            pass
    if isinstance(data, BaseModel):
        return safe_serialize_arbitrary_for_logging(
            data.model_dump(), max_elements, indent
        )
    if dataclasses.is_dataclass(data):
        return safe_serialize_arbitrary_for_logging(
            dataclasses.asdict(data), max_elements, indent
        )
    if isinstance(data, dict):
        keys = list(data.keys())
        result = []
        result.append("{")
        for key in keys[:max_elements]:
            value = _maybe_redacted(key, data[key])
            result.append(
                f'{indent}  {repr(key)}: {safe_serialize_arbitrary_for_logging(value, max_elements, indent + "  ")},'
            )
        if len(keys) > max_elements:
            result.append(f"{indent}  ...,")
        result.append(f"{indent}" + "}")
        return "\n".join(result)
    elif isinstance(data, list):
        result = []
        result.append("[")
        for item in data[:max_elements]:
            result.append(
                f'{indent}  {safe_serialize_arbitrary_for_logging(item, max_elements, indent + "  ")},'
            )
        if len(data) > max_elements:
            _len = len(data)
            result.append(f"{indent}  ...(len={_len}),")
        result.append(f"{indent}" + "]")
        return "\n".join(result)
    elif isinstance(data, str):
        _len = len(data)
        return repr(
            data[:max_elements]
            + (f"...(len={_len})" if _len > max_elements else "")
        )
    elif data is None or isinstance(data, (int, float, bool)):
        return repr(data)
    else:
        return f"<<non-serializable: {type(data).__qualname__}>>"


def to_json(
    file: Union[str, TextIO], max_elements: Optional[int] = None
) -> Callback:
    def _write(event: CallbackEvent, run_id: str, file: TextIO) -> int:
        data_to_serialize = dict(run_id=run_id, evnet=event)
        data_to_write = safe_serialize_arbitrary_for_logging(
            data_to_serialize, max_elements=max_elements
        )
        return file.write(data_to_write)
        # data_write = dict(run_id=run_id, **data_event)
        # return file.write("\n" + safe_serialize_json(data_write, indent=2))

    async def _callback(
        event: CallbackEvent, run_id: str
    ) -> Optional[CallbackResult]:
        if isinstance(file, str):
            with open(file, "a+") as f:
                result = _write(event, run_id, f)
                return CallbackResult(result=result)
        else:
            result = _write(event, run_id, file)
            return CallbackResult(result=result)

    return _callback


def make_run_id() -> str:
    return uuid4().hex
