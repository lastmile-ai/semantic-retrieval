import asyncio
from typing import Protocol


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
