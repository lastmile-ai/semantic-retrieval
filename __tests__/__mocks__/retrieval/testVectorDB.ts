import {
  VectorDB,
  VectorDBQuery,
} from "../../../src/data-store/vector-DBs/vectorDB";
import { Document } from "../../../src/document/document";
import { InMemoryDocumentMetadataDB } from "../../../src/document/metadata/InMemoryDocumentMetadataDB";
import { DocumentMetadataDB } from "../../../src/document/metadata/documentMetadataDB";
import { VectorEmbedding } from "../../../src/transformation/embeddings/embeddings";
import { TestEmbeddings } from "../transformation/embeddings/testEmbeddings";
import {
  AddDocumentsToVectorDBEvent,
  QueryVectorDBEvent,
} from "../../../src/utils/callbacks";

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
    const event: AddDocumentsToVectorDBEvent = {
      name: "onAddDocumentsToVectorDB",
      documents: _documents,
    };

    // Open Q: Should this be await'ed
    this.callbackManager?.runCallbacks(event);
  }

  async query(_query: VectorDBQuery): Promise<VectorEmbedding[]> {
    // Should use mockQuery instead
    const vectorEmbeddings = new Array<VectorEmbedding>();

    const event: QueryVectorDBEvent = {
      name: "onQueryVectorDB",
      query: _query,
      vectorEmbeddings,
    };

    // Open Q: Should this be await'ed
    this.callbackManager?.runCallbacks(event);

    return vectorEmbeddings;
  }
}
