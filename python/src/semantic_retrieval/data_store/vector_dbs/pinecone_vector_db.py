from dataclasses import dataclass
from typing import List, Optional

from semantic_retrieval.data_store.vector_dbs.vector_db import VectorDB, VectorDBConfig

from semantic_retrieval.document.document import Document


def getEnvVar(k):
    # TODO
    return "VALUE"


@dataclass
class PineconeVectorDBConfig(VectorDBConfig):
    index_name: str
    api_key: Optional[str] = None
    environment: Optional[str] = None
    namespace: Optional[str] = None


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
