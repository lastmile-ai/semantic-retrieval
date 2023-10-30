import json
from typing import Dict, List, NewType, Optional
from semantic_retrieval.document.metadata.document_metadata import DocumentMetadata

from semantic_retrieval.document.metadata.document_metadata_db import (
    DocumentMetadataDB,
    DocumentMetadataQuery,
)


DocumentMetadataMap = NewType("DocumentMetadataMap", Dict[str, DocumentMetadata])
# DocumentMetadataMap = Dict[str, DocumentMetadata]


class InMemoryDocumentMetadataDB(DocumentMetadataDB):
    def __init__(self, metadata: Optional[DocumentMetadataMap] = None):
        self.metadata = metadata or {}

    async def get_metadata(self, document_id: str) -> Optional[DocumentMetadata]:
        try:
            return self.metadata.get(document_id)
        except KeyError:
            return None

    async def set_metadata(self, document_id: str, metadata: DocumentMetadata):
        self.metadata[document_id] = metadata

    async def query_document_ids(self, query: DocumentMetadataQuery) -> List[str]:
        return [
            document_id
            for document_id, metadata in self.metadata.items()
            if metadata.metadata.get(query.metadata_key) == query.metadata_value
        ]

    async def persist(self, file_path: str):
        with open(file_path, "w") as file:
            file.write(
                json.dumps({d_id: dmd.model_dump_json() for d_id, dmd in self.metadata.items()})
            )

    @staticmethod
    async def from_json_file(file_path: str) -> "InMemoryDocumentMetadataDB":
        with open(file_path, "r") as file:
            json_data = file.read()
            the_map_ser: Dict[str, str] = json.loads(json_data)
            the_map: DocumentMetadataMap = DocumentMetadataMap(
                {
                    d_id: DocumentMetadata.model_validate_json(dmd_ser)
                    for d_id, dmd_ser in the_map_ser.items()
                }
            )
            print(f"[45]{the_map=}")
            return InMemoryDocumentMetadataDB(the_map)
