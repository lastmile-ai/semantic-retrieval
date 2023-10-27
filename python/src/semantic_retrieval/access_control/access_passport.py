from semantic_retrieval.utils.callbacks import (
    CallbackManager,
    GetAccessIdentityEvent,
    RegisterAccessIdentityEvent,
    Traceable,
)

from semantic_retrieval.access_control.access_identity import (
    AccessIdentity,
)


class AccessPassport(Traceable):
    def __init__(self, callback_manager=None):
        self.access_identities = {}
        self.callback_manager = callback_manager

    def register(self, access_identity):
        self.access_identities.update({access_identity.id: access_identity})

        if self.callback_manager:
            event = RegisterAccessIdentityEvent(
                name="onRegisterAccessIdentity", access_identity=access_identity
            )
            self.callback_manager.run_callbacks(event)

    def get_identity(self, resource):
        access_identity = self.access_identities.get(resource)

        if self.callback_manager:
            event = GetAccessIdentityEvent(
                name="onGetAccessIdentity", access_identity=access_identity
            )
            self.callback_manager.run_callbacks(event)

        return access_identity
