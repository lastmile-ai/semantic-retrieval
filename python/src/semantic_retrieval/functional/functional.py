from typing import List, Tuple, TypeVar

from result import Err, Ok, Result

import functools
from typing import Callable, Concatenate, Generator, TypeVar

from result import Err, Ok, Result

from typing import TypeVar, ParamSpec

PS = ParamSpec("PS")

TR = TypeVar("TR", covariant=True)
E = TypeVar("E", covariant=True)


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


def result_do(
    func: Callable[Concatenate[PS], Generator[Result[TR, E], TR, Result[TR, E]]]
) -> Callable[PS, Result[TR, E]]:
    """
    Example:
    def get_result(success: bool, i: int) -> Result[int, str]:
    if success:
        out = Ok(42 + i)
        # print(f"{out=}")
        return out
    else:
        out = Err("magnets" + str(i))
        return out


    def get_my_thing() -> int:
        return 3


    @result_do
    def result_sum(
        is1: bool, is2: bool
    ) -> DoResult[int, str]:
        r1, r2 = get_result(is1, 1), get_result(is2, 2)
        x1 = yield r1
        x2 = yield r2

        some_other_thing = get_my_thing()
        return Ok(x1 + x2 + some_other_thing)

    """

    @functools.wraps(func)
    def wrapper(*args: PS.args, **kwargs: PS.kwargs) -> Result[TR, E]:
        generator = func(*args, **kwargs)
        try:
            value = next(generator)
            while True:
                match value:
                    case Ok(value_):
                        value = generator.send(value_)
                    case Err():
                        return value
        except StopIteration as e:
            return e.value

    return wrapper
