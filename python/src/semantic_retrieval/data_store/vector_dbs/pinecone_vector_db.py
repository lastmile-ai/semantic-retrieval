from uuid import uuid4 as uuid
from typing import Any, List

from semantic_retrieval.data_store.vector_dbs.vector_db import VectorDB, VectorDBConfig

from semantic_retrieval.document.document import Document


def getEnvVar(k):
    # TODO
    return "VALUE"


class PineconeVectorDB(VectorDB):
    def __init__(self, config: VectorDBConfig):
        # TODO
        pass

    @classmethod
    def from_documents(cls, documents: List[Document], config: VectorDBConfig):
        instance = cls(config)
        instance.add_documents(documents)
        return instance

    def sanitize_metadata(self, unsanitized_metadata):
        # TODO
        pass

    def add_documents(self, documents):
        # TODO
        pass

    def query(self, query):
        # TODO
        pass
