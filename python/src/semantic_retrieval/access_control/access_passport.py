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
        self.access_identities[access_identity.resource] = access_identity
        event = RegisterAccessIdentityEvent("onRegisterAccessIdentity", access_identity)
        if self.callback_manager:
            self.callback_manager.run_callbacks(event)

    def get_identity(self, resource):
        access_identity = self.access_identities.get(resource)
        event = GetAccessIdentityEvent("onGetAccessIdentity", resource, access_identity)
        if self.callback_manager:
            self.callback_manager.run_callbacks(event)
        return access_identity
