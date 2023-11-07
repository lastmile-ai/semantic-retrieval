import TestVectorDB from "../__mocks__/retrieval/testVectorDB";

import { InMemoryDocumentMetadataDB } from "../../src/document/metadata/inMemoryDocumentMetadataDB";
import { VectorDBDocumentRetriever } from "../../src/retrieval/vector-DBs/vectorDBDocumentRetriever";
import { TEST_VECTOR } from "../__mocks__/transformation/embeddings/testEmbeddings";
import { VectorEmbedding } from "../../src/transformation/embeddings/embeddings";
import {
  VectorDBQuery,
  VectorDBTextQuery,
} from "../../src/data-store/vector-DBs/vectorDB";
import { AccessPassport } from "../../src/access-control/accessPassport";

import { getTestDocument } from "../__utils__/testDocumentUtils";
/*  */
// import from evaluation
import {
  evaluateRetrievers,
  RetrievalEvaluationDataset,
} from "../../src/evaluation/evaluation";

import {
  calculateRetrievedFragmentRecall,
  calculateRetrievedFragmentPrecision,
} from "../../src/evaluation/document";
import { AlwaysAllowAccessPolicy } from "../../src";

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

describe("Retrieval evaluation metrics", () => {
  test("Fragment-level precision and recall for VectorDBDocumentRetriever", async () => {
    const metadataDB = new InMemoryDocumentMetadataDB({
      "test-document-id-A": {
        ...DOCUMENT_A_METADATA,
        metadata: {
          test: "test metadata for document A",
        },
        accessPolicies: [new AlwaysAllowAccessPolicy()],
      },
      "test-document-id-B": {
        ...DOCUMENT_B_METADATA,
        accessPolicies: [new AlwaysAllowAccessPolicy()],
      },
      "test-document-id-C": {
        ...DOCUMENT_C_METADATA,
        metadata: {
          test: "test metadata for document C",
        },
        accessPolicies: [new AlwaysAllowAccessPolicy()],
        attributes: { type: "webpage" },
      },
    });

    const vectorDB = new TestVectorDB(metadataDB);
    const retriever = new VectorDBDocumentRetriever({ vectorDB, metadataDB });

    // fragments returned: Aa, Ab, Ba, Ca
    const query: VectorDBTextQuery = {
      text: "test",
      topK: 5,
    };

    const queryParams = {
      accessPassport: new AccessPassport(),
      query,
    };

    const relevantFragmentIds: string[] = [
      "test-fragment-id-Ab",
      "test-fragment-id-Ca",
      "test-fragment-id-Dc",
      "test-fragment-id-Eb",
      "test-fragment-id-Ec",
    ];

    const metrics = [
      calculateRetrievedFragmentRecall,
      calculateRetrievedFragmentPrecision,
    ];

    const evalDataset: RetrievalEvaluationDataset<string[]> = {
      relevantDataByQuery: [[queryParams, relevantFragmentIds]],
    };

    const evalRes = await evaluateRetrievers([retriever], evalDataset, metrics);

    const recall = evalRes[0]["documentFragmentRecall"];
    const precision = evalRes[0]["documentFragmentPrecision"];

    expect(recall).toBe(0.4);
    expect(precision).toBe(0.5);
  });
});
