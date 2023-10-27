from semantic_retrieval.document.metadata.document_metadata import DocumentMetadata
from semantic_retrieval.document.metadata.document_metadata_db import DocumentMetadataDB
from typing import Awaitable, Dict, Optional
import json


class InMemoryDocumentMetadataDB(DocumentMetadataDB):
    def __init__(self, metadata: Optional[Dict[str, DocumentMetadata]] = None):
        self.metadata = metadata if metadata is not None else {}

    async def getMetadata(self, document_id: str) -> DocumentMetadata:
        self.metadata.get(document_id)

    async def setMetadata(self, document_id: str, metadata: DocumentMetadata) -> None:
        self.metadata.update({document_id: metadata})

    async def persist(self, file_path: str) -> None:
        with open(file_path, "w") as f:
            f.write(json.dumps(self.metadata))

    def from_json_file(file_path):
        Awaitable[InMemoryDocumentMetadataDB]
        with open(file_path, "r") as f:
            metadata = f.read()
            map = json.loads(metadata)
            return InMemoryDocumentMetadataDB(map)
