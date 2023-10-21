import TestVectorDB from "../__mocks__/retrieval/testVectorDB";

import { InMemoryDocumentMetadataDB } from "../../src/document/metadata/InMemoryDocumentMetadataDB";
import { VectorDBDocumentRetriever } from "../../src/retrieval/vector-DBs/vectorDBDocumentRetriever";
import { TEST_VECTOR } from "../__mocks__/transformation/embeddings/testEmbeddings";
import { VectorEmbedding } from "../../src/transformation/embeddings/embeddings";
import { VectorDBTextQuery } from "../../src/data-store/vector-DBs/vectorDB";
import { AccessPassport } from "../../src/access-control/accessPassport";
import { ResourceAccessPolicy } from "../../src/access-control/resourceAccessPolicy";
import {
  getTestDocument,
  getTestDocumentFragment,
} from "../utils/testDocumentUtils";
import { AlwaysAllowAccessPolicy } from "../../src/access-control/policies/alwaysAllowAccessPolicy";

import {
  calculateRecall,
  calculatePrecision,
  evaluateDocumentListRetrievalWithGT,
} from "../../src/evaluation/evaluation";

const mockQuery = jest.fn();

jest.mock("../__mocks__/retrieval/testVectorDB", () =>
  jest.fn().mockImplementation(() => ({
    addDocuments: jest.fn(),
    query: mockQuery,
  }))
);

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

const alwaysDenyPolicy: ResourceAccessPolicy = {
  policy: "always-deny",
  testDocumentReadPermission: async () => false,
  testPolicyPermission: async () => false,
};

mockQuery.mockImplementation(async () => retrievedEmbeddings);

const DOCUMENT_A_METADATA = {
  documentId: "test-document-id-A",
  uri: "test-uri-A",
  document: getTestDocument({
    documentId: "test-document-id-A",
    fragments: [],
  }),
  metadata: {},
  attributes: {},
};

const DOCUMENT_B_METADATA = {
  documentId: "test-document-id-B",
  document: getTestDocument({
    documentId: "test-document-id-B",
    fragments: [],
  }),
  uri: "test-uri-B",
  metadata: {},
  attributes: {},
};

const DOCUMENT_C_METADATA = {
  documentId: "test-document-id-C",
  document: getTestDocument({
    documentId: "test-document-id-C",
    fragments: [],
  }),
  uri: "test-uri-C",
  metadata: {},
  attributes: {},
};

describe("retrieveDocuments retrieves correct data", () => {
  test("returns the documents with correct metadata and count", async () => {
    const metadataDB = new InMemoryDocumentMetadataDB({
      "test-document-id-A": {
        ...DOCUMENT_A_METADATA,
        metadata: {
          test: "test metadata for document A",
        },
      },
      "test-document-id-C": {
        ...DOCUMENT_C_METADATA,
        metadata: {
          test: "test metadata for document C",
        },
        attributes: { type: "webpage" },
      },
    });

    const vectorDB = new TestVectorDB(metadataDB);
    const retriever = new VectorDBDocumentRetriever({ vectorDB, metadataDB });

    const query: VectorDBTextQuery = {
      text: "test",
      topK: 5,
    };

    const retrievedDocuments = await retriever.retrieveData({
      accessPassport: new AccessPassport(),
      query,
    });

    // expect(retrievedDocuments.length).toBe(3);
    const relevantIds = ["test-document-id-B", "test-document-id-D"];
    const recall = evaluateDocumentListRetrievalWithGT(
      retrievedDocuments,
      relevantIds,
      calculateRecall
    );
    const precision = evaluateDocumentListRetrievalWithGT(
      retrievedDocuments,
      relevantIds,
      calculatePrecision
    );
    expect(recall).toBe(0.5);
    expect(precision).toBe(1 / 3);
  });
});
