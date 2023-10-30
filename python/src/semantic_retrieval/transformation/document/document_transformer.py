from typing import List, Optional, Sequence

from semantic_retrieval.document.document import Document
from semantic_retrieval.document.metadata.document_metadata_db import DocumentMetadataDB

from semantic_retrieval.transformation.transformer import Transformer

from semantic_retrieval.utils.callbacks import CallbackManager, Traceable


class DocumentTransformer(Transformer):
    pass


class BaseDocumentTransformer(DocumentTransformer, Traceable):
    def __init__(
        self,
        documentMetadataDB: Optional[DocumentMetadataDB] = None,
        callback_manager: Optional[CallbackManager] = None,
    ):
        self.documentMetadataDB = documentMetadataDB
        self.callback_manager = callback_manager

    async def transform_document(self, document: Document) -> Document:
        raise NotImplementedError("This method must be implemented in a derived class")

    async def transform_documents(self, documents: Sequence[Document]) -> List[Document]:
        # TODO
        return []
