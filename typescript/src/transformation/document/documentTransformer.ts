import { Document } from "../../document/document";
import { DocumentMetadataDB } from "../../document/metadata/documentMetadataDB";
import {
  CallbackManager,
  Traceable,
  TransformDocumentsEvent,
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

      // Pre-transformed doc may have source documents from previous transformations
      let sourceDocumentIds = originalDocumentMetadata?.sourceDocumentIds;
      if (
        (sourceDocumentIds == null || sourceDocumentIds.length === 0) &&
        originalDocumentMetadata?.rawDocument != null
      ) {
        // Pre-transformed doc was ingested from a raw document so it is the 'source' document
        sourceDocumentIds = [document.documentId];
      }

      await this.documentMetadataDB?.setMetadata(
        transformedDocument.documentId,
        {
          ...originalDocumentMetadata, // TODO: Probably only need a subset (e.g. accessPolicies), excluding rawDocument
          sourceDocumentIds,
          documentId: transformedDocument.documentId,
          document: transformedDocument,
          uri: originalDocumentMetadata?.uri ?? transformedDocument.documentId,
          metadata: {
            ...originalDocumentMetadata?.metadata,
            ...transformedDocument.metadata,
            transformer: this.constructor.name,
            originalDocumentId: document.documentId,
          },
        }
      );
      return transformedDocument;
    });

    const transformedDocuments = (await Promise.all(transformPromises)).flat();

    const event: TransformDocumentsEvent = {
      name: "onTransformDocuments",
      transformedDocuments,
      originalDocuments: documents,
    };
    await this.callbackManager?.runCallbacks(event);

    return transformedDocuments;
  }
}
