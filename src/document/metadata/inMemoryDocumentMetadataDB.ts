import { DocumentMetadata } from "./documentMetadata";
import {
  DocumentMetadataDB,
  DocumentMetadataQuery,
} from "./documentMetadataDB";
import fs from "fs/promises";

type DocumentMetadataMap = { [key: string]: DocumentMetadata };

export class InMemoryDocumentMetadataDB implements DocumentMetadataDB {
  private metadata: DocumentMetadataMap = {};

  constructor(metadata?: DocumentMetadataMap) {
    this.metadata = metadata ?? this.metadata;
  }

  async getMetadata(documentId: string): Promise<DocumentMetadata> {
    return this.metadata[documentId];
  }

  async setMetadata(
    documentId: string,
    metadata: DocumentMetadata
  ): Promise<void> {
    this.metadata[documentId] = metadata;
  }

  async queryDocumentIds(query: DocumentMetadataQuery): Promise<string[]> {
    return Object.keys(this.metadata).filter((documentId) => {
      const metadata = this.metadata[documentId];
      const metadataValue = metadata.metadata?.[query.metadataKey];
      if (!metadataValue) return false;

      if (query.matchType === "exact") {
        return metadataValue === query.metadataValue;
      } else {
        return metadataValue.includes(query.metadataValue);
      }
    });
  }

  async persist(filePath: string) {
    await fs.writeFile(
      filePath,
      JSON.stringify(this.metadata, (key, value) => {
        // Don't serialize fragment relationships to avoid circular references
        if (key === "previousFragment" || key === "nextFragment") {
          return;
        }
        return value;
      })
    );
  }

  static async fromJSONFile(
    filePath: string
  ): Promise<InMemoryDocumentMetadataDB> {
    const json = await (await fs.readFile(filePath)).toString();
    const map = JSON.parse(json);
    return new InMemoryDocumentMetadataDB(map);
  }
}
