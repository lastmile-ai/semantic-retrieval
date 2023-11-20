import json
import logging
import os
import time
from typing import (
    IO,
    Any,
    Callable,
    Iterable,
    List,
    Mapping,
    Optional,
    ParamSpec,
    TypeVar,
    cast,
)

from jsoncomment import JsonComment
from result import Err, Ok, Result
from semantic_retrieval.common.json_types import JSONObject
from semantic_retrieval.functional.functional import ErrWithTraceback

P = ParamSpec("P")


F = TypeVar("F", bound=Callable[..., Any])
U_Callable = TypeVar("U_Callable", bound=Callable[..., Any])


LOGGER_FMT = "[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d: %(message)s"

LOGGER = logging.getLogger(__name__)
logging.basicConfig(format=LOGGER_FMT)

T = TypeVar("T")
T2 = TypeVar("T2")
U2 = TypeVar("U2")
U = TypeVar("U")

E = TypeVar("E")

Thunk = Callable[[], T]
ResultThunk = Callable[[], Result[T, E]]

G = TypeVar("G", bound=Callable[..., Any])
Decorator = Callable[[F], G]


def text_file_read_with_handle(f_handle: IO[str]) -> Result[str, str]:
    try:
        return Ok(f_handle.read())
    except IOError as e:
        return ErrWithTraceback(e)


def make_text_path_handler(
    f_handle_fn: Callable[[IO[str]], Result[T, str]],
    mode: str,
    encoding: str | None = None,
) -> Callable[[str | None], Result[T, str]]:
    encoding = encoding or "utf8"

    def text_path_handler(path: str | None) -> Result[T, str]:
        if not path:
            return Err("Path is None")
        try:
            with open(path, mode, encoding=encoding) as f:
                return f_handle_fn(f)
        except IOError as e:
            return ErrWithTraceback(e)

    return text_path_handler


def file_contents(path: str | None) -> Result[str, str]:
    path_fn = make_text_path_handler(text_file_read_with_handle, "r")
    return path_fn(path)


def text_file_write(path: str | None, contents: str) -> Result[int, str]:
    def _write(f_handle: IO[str]) -> Result[int, str]:
        return text_file_write_with_handle(f_handle, contents)

    path_fn = make_text_path_handler(_write, "w")
    return path_fn(path)


def text_file_write_with_handle(f_handle: IO[str], contents: str) -> Result[int, str]:
    try:
        return Ok(f_handle.write(contents))
    except IOError as e:
        return ErrWithTraceback(e)


def flatten_list(list_of_lists: List[List[T]]) -> List[T]:
    return [item for sublist in list_of_lists for item in sublist]


def unflatten_iterable(it: Iterable[T], chunk_size: int) -> List[List[T]]:
    out = [[]]
    for x in it:
        if len(out[-1]) == chunk_size:
            out.append([])
        out[-1].append(x)

    return out


def exp_backoff(
    max_retries: int,
    base_delay: int,
    logger: Optional[logging.Logger] = None,
) -> Decorator[Thunk[U], ResultThunk[U, str]]:
    logger = logger or logging.getLogger(__name__)

    def dec(thunk: Thunk[U]) -> ResultThunk[U, str]:
        def wrapper_thunk() -> Result[U, str]:
            retries = 0
            while retries < max_retries:
                try:
                    return Ok(thunk())
                except Exception as e:
                    logger.info(f"Attempt {retries + 1} failed: {e}")
                    delay = base_delay * 2**retries
                    logger.info(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                    retries += 1

            return Err("Max retries reached, operation failed.")

        return wrapper_thunk

    return dec


def remove_nones(d: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


def dict_union_best_effort(
    *dicts: Mapping[T, Any], on_conflict: Optional[str] = None
) -> dict[T, Any]:
    assert on_conflict != "err"
    res = dict_union(*dicts, on_conflict=on_conflict)
    match res:
        case Ok(d):
            return d
        case Err(e):
            assert False, f"should be unreachable: {e=}"


def dict_union(
    *dicts: Mapping[T, Any], on_conflict: Optional[str] = None
) -> Result[dict[T, Any], str]:
    # TODO make this an enum
    on_conflict = on_conflict or "err"

    result = {}
    for d in dicts:
        for k, v in d.items():
            if k not in result or result[k] == v:
                result[k] = v
            else:
                if on_conflict == "err":
                    return Err(f"Key {k} exists with different values: {v}, {result[k]}.")
                elif on_conflict == "keep_first":
                    pass
                elif on_conflict == "replace":
                    result[k] = v
                else:
                    assert False, f"should be unreachable: invalid {on_conflict=}"

    return Ok(result)


def enforce_unique(iterable: Iterable[T]) -> Result[T, str]:
    iter_ = iter(iterable)
    value = next(iter_)
    for x in iter_:
        if value != x:
            return Err(f"distinct values: {value}, {x}")
    return Ok(value)


only = enforce_unique


def dict_map(
    d: Mapping[T, U],
    key_fn: Optional[Callable[[T, U], T2]] = None,
    value_fn: Optional[Callable[[T, U], U2]] = None,
) -> dict[T2, U2]:
    def key_fn_(k: T, v: U) -> T2:
        if key_fn:
            return key_fn(k, v)
        else:
            return cast(T2, k)

    def value_fn_(k: T, v: U) -> U2:
        if value_fn:
            return value_fn(k, v)
        else:
            return cast(U2, v)

    return {key_fn_(k, v): value_fn_(k, v) for k, v in d.items()}


def dict_invert(d: Mapping[T, U]) -> Result[dict[U, T], str]:
    out = {}
    for k, v in d.items():
        if v in out:
            return Err(f"Not one-to-one: {k=},{v=}, {out[v]=}")
        else:
            out[v] = k

    return Ok(out)


def combine_returncodes(returncodes: Iterable[int]) -> int:
    """
    Return 0 if all 0 or first non-zero return code
    """
    for rc in returncodes:
        if rc != 0:
            return rc

    return 0


def load_json(json_str: str) -> Result[JSONObject, str]:
    parser = JsonComment()
    try:
        return Ok(parser.loads(json_str))
    except json.JSONDecodeError as e:
        return ErrWithTraceback(e)


def load_json_file(file_path: str | None) -> Result[JSONObject, str]:
    contents = file_contents(file_path)
    return contents.and_then(load_json)


def normalize_path(path: str) -> str:
    return os.path.realpath(path)


def deprefix(s: str, pfx: str) -> str:
    if s.startswith(pfx):  # Checks if the string starts with the given prefix
        return s[len(pfx) :]  # If true, returns the string without the prefix
    else:
        return s
