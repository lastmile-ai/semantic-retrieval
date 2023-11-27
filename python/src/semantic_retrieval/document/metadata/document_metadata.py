from typing import Any, Dict, Optional

from semantic_retrieval.common.types import Record
from semantic_retrieval.utils.interop import canonical_field


class DocumentMetadata(Record):
    document_id: str
    uri: str
    metadata: Dict[str, str]
    attributes: Dict[str, str]
    # raw_document: Optional[RawDocument] = None
    # document: Optional[Document] = None
    collection_id: Optional[str] = None
    # TODO [P1]: Fix this because fails at pydantic serialization
    # data_source: Optional[DataSource] = None
    name: Optional[str] = None
    mime_type: Optional[str] = None
    hash: Optional[str] = None

    # access_policies: List[ResourceAccessPolicy] = []
    #
    def to_dict(self) -> Dict[str, Any]:
        return {
            canonical_field(field): value
            for field, value in {
                "document_id": self.document_id,
                "uri": self.uri,
                "metadata": self.metadata,
                "attributes": self.attributes,
                # TODO [P1]: Need to make sure that these also end up being serializable when implemented
                # Assuming that RawDocument, Document, DataSource and ResourceAccessPolicy have to_dict function
                # "raw_document": self.raw_document.to_dict() if self.raw_document else None,
                # "document": self.document.to_dict() if self.document else None,
                "collection_id": self.collection_id,
                # "data_source": self.data_source.to_dict() if self.data_source else None,
                "name": self.name,
                "mime_type": self.mime_type,
                "hash": self.hash,
                # Assuming that ResourceAccessPolicy has to_dict function
                # "access_policies": [ap.model_dump_json() for ap in self.access_policies]
                # if self.access_policies
                # else [],
            }.items()
        }


# # Example usage:
# metadata = DocumentMetadata(
#     document_id="12345",
#     uri="https://example.com/document",
#     metadata={"key1": "value1", "key2": "value2"},
#     attributes={"attr1": "attr_value1"},
# )
