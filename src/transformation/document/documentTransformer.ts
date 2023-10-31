import { Document } from "../../document/document";
import { DocumentMetadataDB } from "../../document/metadata/documentMetadataDB";
import {
  CallbackManager,
  Traceable,
  TranformDocumentsEvent,
} from "../../utils/callbacks";
import { Transformer } from "../transformer";

/**
 * A DocumentTransformer can transform one or more documents into
 * one or more new documents. Common transformations include text
 * chunking and summarization.
 */
export interface DocumentTransformer extends Transformer<Document[]> {}

export abstract class BaseDocumentTransformer
  implements DocumentTransformer, Traceable
{
  documentMetadataDB?: DocumentMetadataDB;
  callbackManager?: CallbackManager;

  constructor(
    documentMetadataDB?: DocumentMetadataDB,
    callbackManager?: CallbackManager
  ) {
    this.documentMetadataDB = documentMetadataDB;
    this.callbackManager = callbackManager;
  }

  abstract transformDocument(document: Document): Promise<Document>;

  async transformDocuments(documents: Document[]): Promise<Document[]> {
    const transformPromises = documents.map(async (document) => {
      const transformedDocument = await this.transformDocument(document);
      const originalDocumentMetadata =
        await this.documentMetadataDB?.getMetadata(document.documentId);
      await this.documentMetadataDB?.setMetadata(
        transformedDocument.documentId,
        {
          ...originalDocumentMetadata,
          documentId: transformedDocument.documentId,
          document: transformedDocument,
          uri: originalDocumentMetadata?.uri ?? transformedDocument.documentId,
          metadata: {
            ...originalDocumentMetadata?.metadata,
            transformer: this.constructor.name,
            originalDocumentId: document.documentId,
          },
        }
      );
      return transformedDocument;
    });

    const transformedDocuments = (await Promise.all(transformPromises)).flat();

    const event: TranformDocumentsEvent = {
      name: "onTransformDocuments",
      transformedDocuments,
      originalDocuments: documents,
    };
    this.callbackManager?.runCallbacks(event);

    return transformedDocuments;
  }
}
