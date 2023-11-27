import pandas as pd
from result import Result
from semantic_retrieval.access_control.access_function import (
    AccessFunction,
    get_data_access_checked,
)
from semantic_retrieval.access_control.access_identity import (
    AuthenticatedIdentity,
)
from semantic_retrieval.common.types import CallbackEvent, Record
from semantic_retrieval.retrieval.retriever import BaseRetriever
from semantic_retrieval.utils.callbacks import CallbackManager, Traceable


class CSVRetrieverQuery(Record):
    # For now, assume a single primary key column
    primary_key_column: str


class CSVRetriever(BaseRetriever[pd.DataFrame, CSVRetrieverQuery], Traceable):
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
        callback_manager: CallbackManager,
    ):
        super().__init__()
        self.file_path = file_path
        self.viewer_identity = viewer_identity
        self.user_access_function = user_access_function
        self.callback_manager = callback_manager

    async def retrieve_data(  # type: ignore
        self,
    ) -> Result[pd.DataFrame, str]:
        def _get_data(file_path: str) -> pd.DataFrame:
            return pd.read_csv(file_path).fillna(-1)

        out = await get_data_access_checked(
            self.file_path,
            self.user_access_function,
            _get_data,
            self.file_path,
            self.viewer_identity.viewer_auth_id,
        )

        await self.callback_manager.run_callbacks(
            CallbackEvent(
                name="csv_retriever_retrieved_data",
                data=dict(
                    file_path=self.file_path,
                    result=out,
                ),
            )
        )

        return out
