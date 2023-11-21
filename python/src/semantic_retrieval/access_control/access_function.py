from typing import Awaitable, Callable, List, Protocol, TypeVar

from result import Err, Ok, Result
from semantic_retrieval.common.types import CallbackEvent
from semantic_retrieval.functional.functional import (
    result_reduce_list_separate,
)
from semantic_retrieval.utils.callbacks import CallbackManager, run_thunk_safe

T = TypeVar("T")
U = TypeVar("U")


class AccessFunction(Protocol):
    async def __call__(
        self, resource_auth_id: str, viewer_auth_id: str
    ) -> bool:  # type: ignore [intentional; this is just a type signature.]
        pass


def always_allow() -> AccessFunction:
    async def fn(resource_auth_id: str, viewer_auth_id: str) -> bool:
        return True

    out: AccessFunction = fn
    return out


async def user_access_check(
    user_check: AccessFunction,
    resource_auth_id: str,
    viewer_auth_id: str,
    timeout: int = 1,
) -> bool:
    async def _thunk() -> bool:
        return await user_check(resource_auth_id, viewer_auth_id)

    res_allow = await run_thunk_safe(_thunk(), timeout)
    return res_allow.unwrap_or(False)


async def get_data_access_checked(
    params: T,
    user_access_function: AccessFunction,
    get_data_unsafe: Callable[[T], U],
    resource_auth_id: str,
    viewer_auth_id: str,
) -> Result[U, str]:
    if await user_access_check(
        user_access_function, resource_auth_id, viewer_auth_id
    ):
        return Ok(get_data_unsafe(params))
    else:
        return Err(f"Access denied: {resource_auth_id=}, {viewer_auth_id=}")


async def get_data_access_checked_list(
    params: T,
    user_access_function: AccessFunction,
    get_data_unsafe: Callable[[T], Awaitable[List[U]]],
    resource_auth_id_fn: Callable[[U], Awaitable[str]],
    viewer_auth_id: str,
    cm: CallbackManager,
):
    data_list_unchecked = await get_data_unsafe(params)
    data_list_checked = [
        await get_data_access_checked(
            item,
            user_access_function,
            lambda x: x,
            await resource_auth_id_fn(item),
            viewer_auth_id,
        )
        for item in data_list_unchecked
    ]

    allowed, denied = result_reduce_list_separate(data_list_checked)

    await cm.run_callbacks(
        CallbackEvent(
            name="get_data_access_checked_list",
            data=dict(
                params=params,
                resource_auth_id_fn=resource_auth_id_fn,
                viewer_auth_id=viewer_auth_id,
                n_allowed=len(allowed),
                n_denied=len(denied),
            ),
        ),
    )
    return allowed
