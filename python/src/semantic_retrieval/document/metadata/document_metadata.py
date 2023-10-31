from typing import Any, Dict, Optional
from semantic_retrieval.common.types import Record


class DocumentMetadata(Record):
    document_id: str
    uri: str
    metadata: Dict[str, str]
    attributes: Dict[str, str]
    # raw_document: Optional[RawDocument] = None
    # document: Optional[Document] = None
    collection_id: Optional[str] = None
    # TODO: Fix this because fails at pydantic serialization
    # data_source: Optional[DataSource] = None
    name: Optional[str] = None
    mime_type: Optional[str] = None
    hash: Optional[str] = None
    # TODO: Fix this because fails at pydantic serialization
    # access_policies: Optional[List[ResourceAccessPolicy]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "uri": self.uri,
            "metadata": self.metadata,
            "attributes": self.attributes,
            # TODO: Need to make sure that these also end up being serializable when implemented
            # Assuming that RawDocument, Document, DataSource and ResourceAccessPolicy have to_dict function
            "raw_document": self.raw_document.to_dict() if self.raw_document else None,  # type: ignore [fixme]
            "document": self.document.to_dict() if self.document else None,  # type: ignore [fixme]
            "collection_id": self.collection_id,
            "data_source": self.data_source.to_dict() if self.data_source else None,  # type: ignore [fixme]
            "name": self.name,
            "mime_type": self.mime_type,
            "hash": self.hash,
            # Assuming that ResourceAccessPolicy has to_dict function
            "access_policies": [ap.to_dict() for ap in self.access_policies]  # type: ignore [fixme]
            if self.access_policies  # type: ignore [fixme]
            else None,
        }


# # Example usage:
# metadata = DocumentMetadata(
#     document_id="12345",
#     uri="https://example.com/document",
#     metadata={"key1": "value1", "key2": "value2"},
#     attributes={"attr1": "attr_value1"},
# )
