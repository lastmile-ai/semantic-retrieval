import json
from typing import Dict, List, NewType, Optional

from result import Err, Ok, Result
from semantic_retrieval.common.types import CallbackEvent
from semantic_retrieval.document.metadata.document_metadata import DocumentMetadata

from semantic_retrieval.document.metadata.document_metadata_db import (
    DocumentMetadataDB,
    DocumentMetadataQuery,
)
from semantic_retrieval.utils.callbacks import CallbackManager, Traceable


DocumentMetadataMap = NewType("DocumentMetadataMap", Dict[str, DocumentMetadata])


class InMemoryDocumentMetadataDB(DocumentMetadataDB, Traceable):
    def __init__(
        self,
        callback_manager: CallbackManager,
        metadata: Optional[DocumentMetadataMap] = None,
    ):
        self.metadata = metadata or {}
        self.callback_manager = callback_manager

    async def get_metadata(self, document_id: str) -> Result[DocumentMetadata, str]:
        if document_id in self.metadata:
            return Ok(self.metadata[document_id])
        else:
            return Err(f"Document ID {document_id} not found in metadata DB")

    async def set_metadata(self, document_id: str, metadata: DocumentMetadata):
        self.metadata[document_id] = metadata

    async def query_document_ids(
        self, query: DocumentMetadataQuery, run_id: str
    ) -> List[str]:
        out = [
            document_id
            for document_id, metadata in self.metadata.items()
            if metadata.metadata.get(query.metadata_key) == query.metadata_value
        ]

        await self.callback_manager.run_callbacks(
            CallbackEvent(
                name="query_document_ids",
                data={
                    "query": query,
                    "result": out,
                },
            )
        )

        return out

    async def persist(self, file_path: str):
        with open(file_path, "w") as file:
            file.write(
                json.dumps(
                    {d_id: dmd.model_dump_json() for d_id, dmd in self.metadata.items()}
                )
            )

    @staticmethod
    async def from_json_file(
        file_path: str,
    ) -> Result["InMemoryDocumentMetadataDB", str]:
        with open(file_path, "r") as file:
            json_data = file.read()
            the_map_ser: Dict[str, str] = json.loads(json_data)
            for k, v in the_map_ser.items():
                if not isinstance(v, str):  # type: ignore this is a hack.
                    the_map_ser[k] = json.dumps(v)

            the_map: DocumentMetadataMap = DocumentMetadataMap(
                {
                    d_id: DocumentMetadata(**json.loads(dmd_ser))
                    for d_id, dmd_ser in the_map_ser.items()
                }
            )
            return Ok(
                InMemoryDocumentMetadataDB(
                    callback_manager=CallbackManager.default(), metadata=the_map
                )
            )
