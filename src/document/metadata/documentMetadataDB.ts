import { DocumentMetadata } from "./documentMetadata.js";

export interface DocumentMetadataDB {
  // TODO: saqadri - implement a Postgres implementation of this interface.
  getMetadata(documentId: string): Promise<DocumentMetadata>;
  setMetadata(documentId: string, metadata: DocumentMetadata): Promise<void>;
}
