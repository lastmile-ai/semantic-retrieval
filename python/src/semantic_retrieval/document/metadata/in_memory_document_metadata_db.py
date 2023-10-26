from semantic_retrieval.document.metadata.document_metadata import DocumentMetadata
from semantic_retrieval.document.metadata.document_metadata_db import DocumentMetadataDB
from typing import Dict, Optional


class InMemoryDocumentMetadataDB(DocumentMetadataDB):  # type: ignore
    def __init__(self, metadata: Optional[Dict[str, DocumentMetadata]] = None):  # type: ignore
        self.metadata = metadata if metadata is not None else {}

    #
    async def getMetadata(self, documentId: str) -> DocumentMetadata:  # type: ignore
        return self.metadata.get(documentId, None)

    async def setMetadata(self, documentId: str, metadata: DocumentMetadata) -> None:  # type: ignore
        self.metadata[documentId] = metadata
