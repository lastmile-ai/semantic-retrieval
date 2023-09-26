import type { Document } from "../../document/document.js";
import { DocumentMetadataDB } from "../../document/metadata/documentMetadataDB.js";

import { VectorDB, VectorDBQuery } from "./vectorDB.js";

// TODO: Can we just thinly wrap langchain here and handle the access control layer?
export class PineconeVectorDB extends VectorDB {
  constructor(metadataDB?: DocumentMetadataDB) {
    super(metadataDB);
  }

  static async fromDocuments(
    documents: Document[],
    metadataDB?: DocumentMetadataDB,
  ): Promise<VectorDB> {
    const instance = new this(metadataDB);
    await instance.addDocuments(documents);
    return instance;
  }

  async addDocuments(_documents: Document[]): Promise<void> {}

  async query(_query: VectorDBQuery): Promise<Document[]> {
    throw new Error("Method not implemented.");

    // Perform query with as much filtering at the vector DB level as possible
  }
}
