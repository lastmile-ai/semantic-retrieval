import { DocumentMetadata } from "./documentMetadata";

export type DocumentMetadataQuery = {
  metadataKey: string;
  metadataValue: string;
  matchType: "exact" | "includes";
};

export interface DocumentMetadataDB {
  // TODO: saqadri - implement a Postgres implementation of this interface.
  getMetadata(documentId: string): Promise<DocumentMetadata>;
  setMetadata(documentId: string, metadata: DocumentMetadata): Promise<void>;

  queryDocumentIds(query: DocumentMetadataQuery): Promise<string[]>;
}
