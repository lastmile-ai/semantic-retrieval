from typing import Optional
from semantic_retrieval.utils.callbacks import (
    CallbackManager,
    Traceable,
)

from semantic_retrieval.access_control.access_identity import (
    AccessIdentity,
)


class AccessPassport(Traceable):
    def __init__(self, callback_manager: Optional[CallbackManager] = None):
        self.access_identities = {}
        self.callback_manager = callback_manager

    def register(self, access_identity: AccessIdentity):
        self.access_identities.update({access_identity.resource: access_identity})

        # TODO callback
        # if self.callback_manager:
        #     event = RegisterAccessIdentityEvent(
        #         name="onRegisterAccessIdentity", access_identity=access_identity
        #     )
        #     self.callback_manager.run_callbacks(event)

    def get_identity(self, resource: str) -> AccessIdentity:  # type: ignore [fixme]
        # TODO impl

        # TODO callback
        # if self.callback_manager:
        #     event = GetAccessIdentityEvent(
        #         name="onGetAccessIdentity", access_identity=access_identity
        #     )
        #     self.callback_manager.run_callbacks(event)
        pass
