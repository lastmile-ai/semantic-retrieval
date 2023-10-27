from semantic_retrieval.document.metadata.document_metadata import DocumentMetadata
from semantic_retrieval.document.metadata.document_metadata_db import DocumentMetadataDB
from typing import Awaitable, Dict, Optional
import json
from json import JSONEncoder


class DocumentMetadataEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, DocumentMetadata):
            return obj.model_dump()
        return super().default(obj)


class InMemoryDocumentMetadataDB(DocumentMetadataDB):
    def __init__(self, metadata: Optional[Dict[str, DocumentMetadata]] = None):
        if metadata:
            self.metadata = {k: DocumentMetadata(**v) for k, v in metadata.items()}
        else:
            self.metadata = {}

    async def get_metadata(self, document_id: str) -> Awaitable[DocumentMetadata]:
        return self.metadata.get(document_id)

    async def set_metadata(self, document_id: str, metadata: DocumentMetadata) -> None:
        self.metadata.update({document_id: metadata})

    def persist(self, file_path: str) -> None:
        with open(file_path, "w") as f:
            f.write(json.dumps(self.metadata, cls=DocumentMetadataEncoder))

    def from_json_file(file_path):
        Awaitable[InMemoryDocumentMetadataDB]
        with open(file_path, "r") as f:
            metadata = f.read()
            map = json.loads(metadata)
            new_obj = InMemoryDocumentMetadataDB(map)
            return new_obj
