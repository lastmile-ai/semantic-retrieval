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
    // const csv = Papa.unparse(
    //   Object.entries(this.metadata).map((documentMetadata) => ({
    //     documentId: documentMetadata[0],
    //     metadata: JSON.stringify(documentMetadata[1], (_key, value) => {
    //       // Drop functions from the persisted metadata since they're not useful
    //       if (typeof value !== "function") {
    //         return value;
    //       }
    //     }),
    //   })),
    //   { header: true }
    // );
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
