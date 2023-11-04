import asyncio
from typing import Awaitable, Callable, List, Protocol, TypeVar

from result import Err, Ok, Result

from semantic_retrieval.common.types import result_reduce_list_separate

T = TypeVar("T")
U = TypeVar("U")


class AccessFunction(Protocol):
    async def __call__(
        self, resource_auth_id: str, viewer_auth_id: str
    ) -> bool:  # type: ignore [intentional; this is just a type signature.]
        pass


async def user_access_check(
    user_check: AccessFunction,
    resource_auth_id: str,
    viewer_auth_id: str,
    timeout: int = 1,
) -> bool:
    try:
        user_coro = user_check(resource_auth_id, viewer_auth_id)
        task = asyncio.create_task(user_coro)
        return await asyncio.wait_for(task, timeout=timeout)
    except BaseException as e:  # type: ignore
        # TODO log
        return False


async def get_data_access_checked(
    params: T,
    user_access_function: AccessFunction,
    get_data_unsafe: Callable[[T], U],
    resource_auth_id: str,
    viewer_auth_id: str,
) -> Result[U, str]:
    if await user_access_check(user_access_function, resource_auth_id, viewer_auth_id):
        return Ok(get_data_unsafe(params))
    else:
        return Err("Access denied")


async def get_data_access_checked_list(
    params: T,
    user_access_function: AccessFunction,
    get_data_unsafe: Callable[[T], List[U]],
    resource_auth_id_fn: Callable[[U], Awaitable[str]],
    viewer_auth_id: str,
):
    the_list = get_data_unsafe(params)
    list_checked = [
        await get_data_access_checked(
            item,
            user_access_function,
            lambda x: x,
            await resource_auth_id_fn(item),
            viewer_auth_id,
        )
        for item in the_list
    ]

    allowed, _ = result_reduce_list_separate(list_checked)
    # TODO [P1]: log denied
    return allowed
