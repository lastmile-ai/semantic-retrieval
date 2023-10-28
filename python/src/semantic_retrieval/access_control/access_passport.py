from semantic_retrieval.utils.callbacks import (
    CallbackManager,
    GetAccessIdentityEvent,
    RegisterAccessIdentityEvent,
    Traceable,
)
import asyncio

from semantic_retrieval.access_control.access_identity import (
    AccessIdentity,
)


class AccessPassport(Traceable):
    def __init__(self, callback_manager: CallbackManager | None = None):
        self.access_identities: dict[str, AccessIdentity] = {}
        if callback_manager is not None:
            self.callback_manager = callback_manager

    def register(self, access_identity: AccessIdentity):
        self.access_identities.update({access_identity.resource: access_identity})

        if self.callback_manager:
            event = RegisterAccessIdentityEvent(
                name="onRegisterAccessIdentity", identity=access_identity
            )
            
            asyncio.run(self.callback_manager.run_callbacks(event))

    def get_identity(self, resource: str) -> AccessIdentity | None:
        access_identity = self.access_identities.get(resource)

        if self.callback_manager and access_identity is not None:
            event = GetAccessIdentityEvent(
                name="onGetAccessIdentity", identity=access_identity
            )
            asyncio.run(self.callback_manager.run_callbacks(event))

        return access_identity
