from typing import List, Dict, Optional

from semantic_retrieval.document.document import Document, RawDocument
from semantic_retrieval.ingestion.data_sources.data_source import DataSource

from semantic_retrieval.access_control.resource_access_policy import ResourceAccessPolicy


class DocumentMetadata:
    documentId: str
    uri: str
    metadata: Dict[str, str]
    attributes: Dict[str, str]
    rawDocument: Optional[RawDocument] = None
    document: Optional[Document] = None
    collectionId: Optional[str] = None
    dataSource: Optional[DataSource] = None
    name: Optional[str] = None
    mimeType: Optional[str] = None
    hash: Optional[str] = None
    accessPolicies: Optional[List[ResourceAccessPolicy]] = None


# # Example usage:
# metadata = DocumentMetadata(
#     documentId="12345",
#     uri="https://example.com/document",
#     metadata={"key1": "value1", "key2": "value2"},
#     attributes={"attr1": "attr_value1"},
# )
