import { DocumentMetadata } from "./documentMetadata";

export interface DocumentMetadataDB {
    // TODO: saqadri - implement an InMemory and a Postgres implementation of this interface.
    getMetadata(documentId: string): Promise<DocumentMetadata>;
    setMetadata(documentId: string, metadata: DocumentMetadata): Promise<void>;
}
  