from typing import List, Tuple, TypeVar

from result import Err, Ok, Result

T = TypeVar("T")


def result_reduce_list_separate(lst: List[Result[T, str]]) -> Tuple[List[T], List[str]]:
    oks, errs = [], []
    for item in lst:
        match item:
            case Ok(x):
                oks.append(x)
            case Err(e):
                errs.append(e)

    return oks, errs
