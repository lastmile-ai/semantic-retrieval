from abc import abstractmethod
from typing import Generic, Optional

from result import Result
from semantic_retrieval.access_control.access_passport import AccessPassport
from semantic_retrieval.common.types import Q, R
from semantic_retrieval.document.metadata.document_metadata_db import (
    DocumentMetadataDB,
)
from semantic_retrieval.utils.callbacks import CallbackManager


class BaseRetrieverQueryParams(Generic[Q]):
    def __init__(self, access_passport: AccessPassport, query: Q) -> None:
        self.access_passport = access_passport
        self.query = query


class BaseRetriever(Generic[R, Q]):
    def __init__(
        self,
        metadata_db: Optional[DocumentMetadataDB] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self.metadata_db = metadata_db
        self.callback_manager = callback_manager

    @abstractmethod
    async def retrieve_data(
        self, params: BaseRetrieverQueryParams[Q]
    ) -> Result[R, str]:
        pass
