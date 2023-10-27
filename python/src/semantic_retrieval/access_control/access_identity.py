from semantic_retrieval.common.base import Attributable


class AccessIdentity(Attributable):
    resource: str

    def __init__(self, resource: str):
        self.resource = resource
