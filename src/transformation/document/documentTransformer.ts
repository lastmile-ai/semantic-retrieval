import { Document } from "../../document/document";
import { DocumentMetadataDB } from "../../document/metadata/documentMetadataDB";
import { Transformer } from "../transformer";

/**
 * A DocumentTransformer can transform one or more documents into
 * one or more new documents. Common transformations include text
 * chunking and summarization.
 */
export interface DocumentTransformer
  extends Transformer<Document[]> {}

export abstract class BaseDocumentTransformer implements DocumentTransformer {
  documentMetadataDB?: DocumentMetadataDB;

  constructor(documentMetadataDB?: DocumentMetadataDB) {
    this.documentMetadataDB = documentMetadataDB;
  }
  
  abstract transformDocument(document: Document): Promise<Document>;

  async transformDocuments(documents: Document[]): Promise<Document[]> {
    const transformPromises = documents.map((document) =>
      this.transformDocument(document)
    );
    return (await Promise.all(transformPromises)).flat();
  }
}
