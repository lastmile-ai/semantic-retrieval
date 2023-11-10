import logging
import time
from typing import Any, Callable, Iterable, List, Optional, ParamSpec, TypeVar

from result import Err, Ok, Result


P = ParamSpec("P")


F = TypeVar("F", bound=Callable[..., Any])
U_Callable = TypeVar("U_Callable", bound=Callable[..., Any])


LOGGER_FMT = "[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d: %(message)s"

logger = logging.getLogger(__name__)
logging.basicConfig(format=LOGGER_FMT)

T = TypeVar("T")
U = TypeVar("U")

E = TypeVar("E")

Thunk = Callable[[], T]
ResultThunk = Callable[[], Result[T, E]]

G = TypeVar("G", bound=Callable[..., Any])
Decorator = Callable[[F], G]


def file_contents(path: str):
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception as e:
        # TODO [P1] deal with this
        logger.critical("exn=" + str(e))
        return ""


def text_file_write(path: str, contents: str):
    try:
        with open(path, "w") as f:
            return f.write(contents)
    except Exception as e:
        # TODO [P1] deal with this
        logger.critical("exn=" + str(e))
        return ""


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
