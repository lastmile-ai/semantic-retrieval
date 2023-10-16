import {
  VectorDB,
  VectorDBQuery,
} from "../../../src/data-store/vector-DBs/vectorDB";
import { Document } from "../../../src/document/document";
import { InMemoryDocumentMetadataDB } from "../../../src/document/metadata/InMemoryDocumentMetadataDB";
import { DocumentMetadataDB } from "../../../src/document/metadata/documentMetadataDB";
import { VectorEmbedding } from "../../../src/transformation/embeddings/embeddings";
import { TestEmbeddings } from "../transformation/embeddings/testEmbeddings";

/**
 * A VectorDB to use for testing with mocked addDocuments and query methods.
 * To use, import TestVectorDB from this file into the jest test file. 
 * Then, mock the TestVectorDB import with the following:
 
 const mockAddDocuments = jest.fn();
 const mockQuery = jest.fn();

 jest.mock("../__mocks__/retrieval/testVectorDB", () =>
  jest.fn().mockImplementation(() => ({
    addDocuments: mockAddDocuments, // optional; no-op by default
    query: mockQuery,
  }))
);
 */
export default class TestVectorDB extends VectorDB {
  constructor(metadataDB?: DocumentMetadataDB) {
    super(new TestEmbeddings(), metadataDB ?? new InMemoryDocumentMetadataDB());
  }

  async addDocuments(_documents: Document[]): Promise<void> {
    // Should use mockAddDocuments instead
  }

  async query(_query: VectorDBQuery): Promise<VectorEmbedding[]> {
    throw new Error("Should use mockQuery instead");
  }
}
