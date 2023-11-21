from abc import abstractmethod
from typing import List, Sequence

from semantic_retrieval.common.types import CallbackEvent
from semantic_retrieval.document.document import Document
from semantic_retrieval.document.metadata.document_metadata import (
    DocumentMetadata,
)
from semantic_retrieval.document.metadata.document_metadata_db import (
    DocumentMetadataDB,
)
from semantic_retrieval.transformation.transformer import Transformer
from semantic_retrieval.utils.callbacks import CallbackManager, Traceable


class DocumentTransformer(Transformer):
    pass


class BaseDocumentTransformer(DocumentTransformer, Traceable):
    def __init__(
        self,
        callback_manager: CallbackManager,
        document_metadata_db: DocumentMetadataDB,
    ):
        self.document_metadata_db = document_metadata_db
        self.callback_manager = callback_manager

    @abstractmethod
    async def transform_document(self, document: Document) -> Document:
        pass

    async def transform_documents(
        self, documents: Sequence[Document]
    ) -> List[Document]:
        out: List[Document] = []
        for document in documents:
            transformed_doc = await self.transform_document(document)
            out.append(transformed_doc)

            res_original_document_metadata = (
                await self.document_metadata_db.get_metadata(
                    document_id=document.document_id
                )
            )
            original_document_metadata = (
                res_original_document_metadata.unwrap()
            )
            metadata_to_write = DocumentMetadata(
                document_id=transformed_doc.document_id,
                # document=document,
                uri=original_document_metadata.uri
                or transformed_doc.document_id,
                metadata=dict(
                    transformer=self.__class__.__name__,
                    original_document_id=document.document_id,
                    # **(original_document_metadata.metadata or {})
                ),
                attributes={},
            )
            d_odm = original_document_metadata.to_dict()
            if not metadata_to_write.uri:
                metadata_to_write.uri = (
                    original_document_metadata.uri
                    or transformed_doc.document_id
                )
            if not metadata_to_write.uri:
                metadata_to_write.uri = transformed_doc.document_id

            md_to_write = metadata_to_write.metadata
            for k, v in md_to_write.items():
                if not v:
                    md_to_write[k] = d_odm.get(k, v)

            _set_transformed_metadata_res = (
                await self.document_metadata_db.set_metadata(
                    transformed_doc.document_id,
                    metadata=metadata_to_write,
                )
            )

        # const transformPromises = documents.map(async (document) => {
        #       const transformedDocument = await this.transformDocument(document);
        #       const originalDocumentMetadata =
        #         await this.documentMetadataDB?.getMetadata(document.documentId);
        #       await this.documentMetadataDB?.setMetadata(
        #         transformedDocument.documentId,
        #         {
        #           ...originalDocumentMetadata,
        #           documentId: transformedDocument.documentId,
        #           document: transformedDocument,
        #           uri: originalDocumentMetadata?.uri ?? transformedDocument.documentId,
        #           metadata: {
        #             ...originalDocumentMetadata?.metadata,
        #             transformer: this.constructor.name,
        #             originalDocumentId: document.documentId,
        #           },
        #         }
        #       );
        #       return transformedDocument;
        #     });

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
