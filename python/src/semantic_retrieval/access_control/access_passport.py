from typing import Optional

from result import Err, Ok, Result
from semantic_retrieval.access_control.access_identity import AccessIdentity
from semantic_retrieval.utils.callbacks import CallbackManager, Traceable


class AccessPassport(Traceable):
    def __init__(self, callback_manager: Optional[CallbackManager] = None):
        self.access_identities = {}
        self.callback_manager = callback_manager

    def register(self, access_identity: AccessIdentity):
        self.access_identities.update(
            {access_identity.resource: access_identity}
        )

    def get_identity(self, resource: str) -> Result[AccessIdentity, str]:
        if resource in self.access_identities:
            return Ok(self.access_identities[resource])
        else:
            return Err("resource not registered")
