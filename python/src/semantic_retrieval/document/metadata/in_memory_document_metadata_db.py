from semantic_retrieval.document.metadata.document_metadata import DocumentMetadata
from semantic_retrieval.document.metadata.document_metadata_db import DocumentMetadataDB
from typing import Awaitable, Dict, Optional, Any
import json
from json import JSONEncoder


class DocumentMetadataEncoder(JSONEncoder):
    def default(self, o: Any):
        if isinstance(o, DocumentMetadata):
            return o.model_dump()
        return super().default(o)


class InMemoryDocumentMetadataDB(DocumentMetadataDB):
    def __init__(self, metadata: Optional[Dict[str, Any]] = None):
        if metadata:
            self.metadata = {k: DocumentMetadata(**v) for k, v in metadata.items()}
        else:
            self.metadata = {}

    async def get_metadata(self, document_id: str) -> DocumentMetadata | None:
        return self.metadata.get(document_id)

    async def set_metadata(self, document_id: str, metadata: DocumentMetadata) -> None:
        self.metadata.update({document_id: metadata})

    def persist(self, file_path: str) -> None:
        with open(file_path, "w") as f:
            f.write(json.dumps(self.metadata, cls=DocumentMetadataEncoder))

    @staticmethod
    def from_json_file(file_path: str):
        Awaitable[InMemoryDocumentMetadataDB]
        with open(file_path, "r") as f:
            metadata = f.read()
            map = json.loads(metadata)
            new_obj = InMemoryDocumentMetadataDB(map)
            return new_obj
