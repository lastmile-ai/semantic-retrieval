import { DocumentMetadata } from "./documentMetadata";
import { DocumentMetadataDB } from "./documentMetadataDB";
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

  async persist(filePath: string) {
    await fs.writeFile(filePath, JSON.stringify(this.metadata));
  }

  static async fromJSONFile(
    filePath: string
  ): Promise<InMemoryDocumentMetadataDB> {
    const json = await (await fs.readFile(filePath)).toString();
    const map = JSON.parse(json);
    return new InMemoryDocumentMetadataDB(map);
  }
}
