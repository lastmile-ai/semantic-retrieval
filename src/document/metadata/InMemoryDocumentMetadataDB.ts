import { DocumentMetadata } from "./documentMetadata.js";
import { DocumentMetadataDB } from "./documentMetadataDB.js";

export class InMemoryDocumentMetadataDB implements DocumentMetadataDB {
  _metadata: { [key: string]: DocumentMetadata } = {};

  constructor() {}

  async getMetadata(documentId: string): Promise<DocumentMetadata> {
    return this._metadata[documentId];
  }

  async setMetadata(
    documentId: string,
    metadata: DocumentMetadata,
  ): Promise<void> {
    this._metadata[documentId] = metadata;
  }
}
