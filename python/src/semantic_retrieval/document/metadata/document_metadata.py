from typing import Dict, Optional, List

# from dataclasses import dataclass
from pydantic import BaseModel
from semantic_retrieval.access_control.resource_access_policy import ResourceAccessPolicy

from semantic_retrieval.document.document import Document, RawDocument



class DocumentMetadata(BaseModel):
    document_id: str
    uri: str
    metadata: Dict[str, str]
    attributes: Dict[str, str]
    raw_document: Optional[RawDocument] = None
    document: Optional[Document] = None
    collection_id: Optional[str] = None
    # TODO: Fix this because fails at pydantic serialization
    # data_source: Optional[DataSource] = None
    name: Optional[str] = None
    mime_type: Optional[str] = None
    hash: Optional[str] = None
    access_policies: Optional[List[ResourceAccessPolicy]] = None


# # Example usage:
# metadata = DocumentMetadata(
#     documentId="12345",
#     uri="https://example.com/document",
#     metadata={"key1": "value1", "key2": "value2"},
#     attributes={"attr1": "attr_value1"},
# )
