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
        # TODO
        pass

    def get_identity(self, resource):
        # TODO
        pass
