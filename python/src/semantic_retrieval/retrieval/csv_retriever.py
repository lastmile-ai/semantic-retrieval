import pandas as pd
from result import Err, Ok, Result
from semantic_retrieval.access_control.access_function import (
    AccessFunction,
    user_access_check,
)
from semantic_retrieval.access_control.access_identity import AuthenticatedIdentity

from semantic_retrieval.common.types import Record
from semantic_retrieval.retrieval.retriever import (
    BaseRetriever,
)


class CSVRetrieverQuery(Record):
    # For now, assume a single primary key column
    primary_key_column: str


class CSVRetriever(BaseRetriever[pd.DataFrame, CSVRetrieverQuery]):
    """
    CSV retriever enforces RBAC at file granularity.
    If there is a need, this could be done at row-level.

    For simplicity, CSV retriever uses file path as the RBAC resource authentication ID.
    """

    def __init__(
        self,
        file_path: str,
        viewer_identity: AuthenticatedIdentity,
        user_access_function: AccessFunction,
    ):
        super().__init__()
        self.file_path = file_path
        self.viewer_identity = viewer_identity
        self.user_access_function = user_access_function

    async def retrieve_data(  # type: ignore
        self,
    ) -> Result[pd.DataFrame, str]:
        if await user_access_check(
            self.user_access_function,
            self.file_path,
            self.viewer_identity.viewer_auth_id,
        ):
            return Ok(_get_data(self.file_path))
        else:
            return Err("Access denied")


def _get_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path).fillna(-1)
