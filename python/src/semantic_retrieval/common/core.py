import logging
from typing import Iterable, List, TypeVar


LOGGER_FMT = "[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d: %(message)s"

logger = logging.getLogger(__name__)
logging.basicConfig(format=LOGGER_FMT)

T = TypeVar("T")


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
