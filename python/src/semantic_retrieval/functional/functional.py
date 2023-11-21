import functools
import traceback
from typing import (
    Callable,
    Concatenate,
    Generator,
    Iterable,
    ParamSpec,
    Tuple,
    TypeVar,
)

from result import Err, Ok, Result

PS = ParamSpec("PS")

TR = TypeVar("TR", covariant=True)
E = TypeVar("E", covariant=True)


T = TypeVar("T")


def ErrWithTraceback(e: Exception, extra_msg: str = "") -> Result[T, str]:
    if extra_msg:
        extra_msg = extra_msg.rstrip(" :\n")
        extra_msg = f"{extra_msg}"

    return Err(f"{extra_msg}\nException:\n{e}\n{traceback.format_exc()}")


def print_result(r: Result[T, str]) -> None:
    match r:
        case Ok(value):
            print(f"Ok:\n" + str(value))
        case Err(msg):
            print(f"Err:\n" + msg)


def result_to_exitcode(r: Result[T, str], fail_code: int = 1) -> int:
    def _ok(_: T) -> int:
        return 0

    return r.map(_ok).unwrap_or(fail_code)


def result_reduce_list_separate(
    lst: Iterable[Result[T, str]]
) -> Tuple[list[T], list[str]]:
    oks, errs = [], []
    for item in lst:
        match item:
            case Ok(x):
                oks.append(x)
            case Err(e):
                errs.append(e)

    return oks, errs


def result_do(
    func: Callable[
        Concatenate[PS], Generator[Result[TR, E], TR, Result[TR, E]]
    ]
) -> Callable[PS, Result[TR, E]]:
    """
    Example:

    import itertools
    from semantic_retrieval.common.types import DoResult

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
    def result_sum(is1: bool, is2: bool) -> DoResult[int, str]:
        # Notice that we do no explicit error handling here.
        r1, r2 = get_result(is1, 1), get_result(is2, 2)
        x1 = yield r1
        x2 = yield r2

        some_other_thing = get_my_thing()
        return Ok(x1 + x2 + some_other_thing)


    if __name__ == "__main__":
        # This runs all 4 combinations of success/failure.
        # The output will be the Ok result or first error.
        for is1, is2 in itertools.product([True, False], repeat=2):
            the_result: Result[int, str] = result_sum(is1, is2)
            print(f"{the_result=}")

        # Output: ```
            the_result=Ok(90)
            the_result=Err('magnets2')
            the_result=Err('magnets1')
            the_result=Err('magnets1')
        ```

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


def result_reduce_list_all_ok(
    lst: Iterable[Result[T, str]]
) -> Result[list[T], str]:
    oks, errs = result_reduce_list_separate(lst)
    if errs:
        return Err("\n".join(errs))
    else:
        return Ok(oks)
