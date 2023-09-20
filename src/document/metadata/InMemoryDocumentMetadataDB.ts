import { DocumentMetadata } from "./documentMetadata";
import { DocumentMetadataDB } from "./documentMetadataDB";

export class InMemoryDocumentMetadataDB implements DocumentMetadataDB {
    _metadata: {[key: string]: DocumentMetadata} = {};

    constructor() {}

    async getMetadata(documentId: string): Promise<DocumentMetadata> {
        return this._metadata[documentId];
    }

    async setMetadata(documentId: string, metadata: DocumentMetadata): Promise<void> {
        this._metadata[documentId] = metadata;
    }
}