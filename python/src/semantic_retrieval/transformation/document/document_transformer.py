from typing import List, Optional
import asyncio

from semantic_retrieval.document.document import Document
from semantic_retrieval.document.metadata.document_metadata_db import DocumentMetadataDB

from semantic_retrieval.transformation.transformer import Transformer

from semantic_retrieval.utils.callbacks import CallbackManager, Traceable, TransformDocumentsEvent


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

    async def transform_documents(self, documents: List[Document]) -> List[Document]:
        transform_promises = [self.transform_document(document) for document in documents]
        transformed_documents = [
            document
            for sublist in await asyncio.gather(*transform_promises)
            for document in sublist
        ]

        event = TransformDocumentsEvent(
            name="onTransformDocuments",
            transformedDocuments=transformed_documents,
            originalDocuments=documents,
        )
        if self.callback_manager:
            self.callback_manager.run_callbacks(event)

        return transformed_documents
