import TestVectorDB from "../__mocks__/retrieval/testVectorDB";

import { InMemoryDocumentMetadataDB } from "../../src/document/metadata/InMemoryDocumentMetadataDB";
import { VectorDBDocumentRetriever } from "../../src/retrieval/vector-DBs/vectorDBDocumentRetriever";
import { TEST_VECTOR } from "../__mocks__/transformation/embeddings/testEmbeddings";
import { VectorEmbedding } from "../../src/transformation/embeddings/embeddings";
import { VectorDBTextQuery } from "../../src/data-store/vector-DBs/vectorDB";
import { AccessPassport } from "../../src/access-control/accessPassport";

const mockQuery = jest.fn();

jest.mock("../__mocks__/retrieval/testVectorDB", () =>
  jest.fn().mockImplementation(() => ({
    addDocuments: jest.fn(),
    query: mockQuery,
  }))
);

const mockGetMetadata = jest.fn();

jest.mock("../../src/document/metadata/InMemoryDocumentMetadataDB", () => {
  return {
    InMemoryDocumentMetadataDB: jest.fn().mockImplementation(() => ({
      getMetadata: mockGetMetadata,
    })),
  };
});

const metadataDB = new InMemoryDocumentMetadataDB();

const retrievedEmbeddings: VectorEmbedding[] = [
  {
    vector: TEST_VECTOR,
    text: "Test fragment 1 text",
    metadata: {
      source: "test-source-A",
      stringArray: ["this", "is", "fine"],
      num: 1,
      bool: true,
      documentId: "test-document-id-A",
      fragmentId: "test-fragment-id-Aa",
      nested: {
        nestedString: "nested",
        nestedNum: 2,
        nestedBool: false,
        nestedStringArray: ["all", "good"],
        doubleNested: {
          val: "test",
        },
      },
      retrievalScore: 0.9,
    },
    attributes: {},
  },
  {
    vector: TEST_VECTOR,
    text: "Test fragment 2 text",
    metadata: {
      source: "test-source-A",
      documentId: "test-document-id-A",
      fragmentId: "test-fragment-id-Ab",
      retrievalScore: 0.8,
    },
    attributes: {},
  },
  {
    vector: TEST_VECTOR,
    text: "Test fragment 3 text",
    metadata: {
      source: "test-source-B",
      documentId: "test-document-id-B",
      fragmentId: "test-fragment-id-Ba",
      retrievalScore: 0.7,
    },
    attributes: {},
  },
  {
    vector: TEST_VECTOR,
    text: "Test fragment 4 text",
    metadata: {
      source: "test-source-C",
      documentId: "test-document-id-C",
      fragmentId: "test-fragment-id-Ca",
      retrievalScore: 0.6,
    },
    attributes: {},
  },
];

mockQuery.mockImplementation(async () => retrievedEmbeddings);

const vectorDB = new TestVectorDB(metadataDB);
const retriever = new VectorDBDocumentRetriever({ vectorDB, metadataDB });

describe("retrieveDocuments", () => {
  test("returns the documents with correct metadata and count", async () => {
    const query: VectorDBTextQuery = {
      text: "test",
      topK: 5,
    };

    const retrievedDocuments = await retriever.retrieveData({
      accessPassport: new AccessPassport(),
      query,
    });

    expect(retrievedDocuments.length).toBe(3);
    expect(retrievedDocuments[0].fragments.length).toBe(2);
    expect(retrievedDocuments[1].fragments.length).toBe(1);
    expect(retrievedDocuments[2].fragments.length).toBe(1);
  });
});
