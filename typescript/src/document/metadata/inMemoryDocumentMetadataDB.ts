import { DocumentMetadata } from "./documentMetadata";
import { Document } from "../document";
import {
  DocumentMetadataDB,
  DocumentMetadataDBQuery,
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

  async queryDocumentIds(query: DocumentMetadataDBQuery): Promise<string[]> {
    return Object.keys(this.metadata).filter((documentId) => {
      const metadata = this.metadata[documentId];

      switch (query.type) {
        case "metadata": {
          const metadataValue = metadata.metadata?.[query.metadataKey];
          if (!metadataValue) return false;

          if (query.matchType === "exact") {
            return metadataValue === query.metadataValue;
          } else {
            return metadataValue.includes(query.metadataValue);
          }
        }

        case "string_field": {
          const fieldValue = metadata[query.fieldName];
          if (!fieldValue) return false;

          if (query.matchType === "exact") {
            return fieldValue === query.fieldValue;
          } else {
            return fieldValue.includes(query.fieldValue);
          }
        }
      }
    });
  }

  async persist(filePath: string, options?: { persistFragments: boolean }) {
    await fs.writeFile(
      filePath,
      JSON.stringify(
        this.metadata,
        (key, value) => {
          // Don't serialize fragment relationships to avoid circular references
          if (key === "previousFragment" || key === "nextFragment") {
            return;
          }
          if (key === "callbacks") {
            return;
          }
          if (!options?.persistFragments && key === "fragments") {
            return [];
          }
          return value;
        },
        2
      )
    );
  }

  static async fromJSONFile(
    filePath: string,
    deserializer?: (key: string, value: unknown) => unknown
  ): Promise<InMemoryDocumentMetadataDB> {
    const json = await (await fs.readFile(filePath)).toString();
    const map = JSON.parse(json, deserializer);
    (Object.values(map) as DocumentMetadata[]).forEach((metadata) => {
      if (!metadata.document) {
        metadata.document = {
          documentId: metadata.documentId,
          collectionId: metadata.collectionId,
          fragments: [],
          serialize: async () => {
            throw new Error(
              "Unable to serialize document with partial data from metadataDB"
            );
          },
        } as Document;
      }
    });
    return new InMemoryDocumentMetadataDB(map);
  }
}
