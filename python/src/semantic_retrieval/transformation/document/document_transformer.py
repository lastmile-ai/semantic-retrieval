from abc import abstractmethod
from typing import List, Optional, Sequence
from semantic_retrieval.common.types import CallbackEvent

from semantic_retrieval.document.document import Document
from semantic_retrieval.document.metadata.document_metadata_db import DocumentMetadataDB

from semantic_retrieval.transformation.transformer import Transformer

from semantic_retrieval.utils.callbacks import CallbackManager, Traceable


class DocumentTransformer(Transformer):
    pass


class BaseDocumentTransformer(DocumentTransformer, Traceable):
    def __init__(
        self,
        callback_manager: CallbackManager,
        documentMetadataDB: Optional[DocumentMetadataDB] = None,
    ):
        self.documentMetadataDB = documentMetadataDB
        self.callback_manager = callback_manager

    @abstractmethod
    async def transform_document(self, document: Document) -> Document:
        pass

    async def transform_documents(
        self, documents: Sequence[Document]
    ) -> List[Document]:
        out = [await self.transform_document(document) for document in documents]
        await self.callback_manager.run_callbacks(
            CallbackEvent(
                name="transform_documents",
                data={
                    "documents": documents,
                    "result": out,
                },
            )
        )
        return out
