import TestVectorDB from "../__mocks__/retrieval/testVectorDB";

import { InMemoryDocumentMetadataDB } from "../../src/document/metadata/inMemoryDocumentMetadataDB";
import { VectorDBDocumentRetriever } from "../../src/retrieval/vector-DBs/vectorDBDocumentRetriever";
import { TEST_VECTOR } from "../__mocks__/transformation/embeddings/testEmbeddings";
import { VectorEmbedding } from "../../src/transformation/embeddings/embeddings";
import { VectorDBTextQuery } from "../../src/data-store/vector-DBs/vectorDB";
import { AccessPassport } from "../../src/access-control/accessPassport";
import { ResourceAccessPolicy } from "../../src/access-control/resourceAccessPolicy";
import { getTestDocument } from "../__utils__/testDocumentUtils";
import { AlwaysAllowAccessPolicy } from "../../src/access-control/policies/alwaysAllowAccessPolicy";
import { AlwaysDenyAccessPolicy } from "../../src/access-control/policies/alwaysDenyAccessPolicy";

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

const alwaysDenyPolicy = new AlwaysDenyAccessPolicy();

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
        attributes: { type: "webpage" },
        accessPolicies: [new AlwaysAllowAccessPolicy()],
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

    expect(retrievedDocuments.length).toBe(3);
    expect(retrievedDocuments[0].fragments.length).toBe(2);
    expect(retrievedDocuments[1].fragments.length).toBe(1);
    expect(retrievedDocuments[2].fragments.length).toBe(1);

    expect(retrievedDocuments[0].fragments[0].fragmentId).toBe(
      "test-fragment-id-Aa"
    );
    expect(retrievedDocuments[0].fragments[0].documentId).toBe(
      "test-document-id-A"
    );
    expect(retrievedDocuments[0].fragments[0].metadata).toEqual({
      source: "test-source-A",
      stringArray: ["this", "is", "fine"],
      num: 1,
      bool: true,
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
    });

    expect(retrievedDocuments[0].fragments[1].fragmentId).toBe(
      "test-fragment-id-Ab"
    );
    expect(retrievedDocuments[0].fragments[1].documentId).toBe(
      "test-document-id-A"
    );
    expect(retrievedDocuments[0].fragments[1].metadata).toEqual({
      source: "test-source-A",
      retrievalScore: 0.8,
    });

    expect(retrievedDocuments[0].documentId).toBe("test-document-id-A");
    expect(retrievedDocuments[0].metadata).toEqual({
      test: "test metadata for document A",
    });

    expect(retrievedDocuments[1].fragments[0].fragmentId).toBe(
      "test-fragment-id-Ba"
    );
    expect(retrievedDocuments[1].fragments[0].documentId).toBe(
      "test-document-id-B"
    );
    expect(retrievedDocuments[1].fragments[0].metadata).toEqual({
      source: "test-source-B",
      retrievalScore: 0.7,
    });
    expect(retrievedDocuments[1].documentId).toBe("test-document-id-B");
    expect(retrievedDocuments[1].metadata).toEqual({});

    expect(retrievedDocuments[2].fragments[0].fragmentId).toBe(
      "test-fragment-id-Ca"
    );
    expect(retrievedDocuments[2].fragments[0].documentId).toBe(
      "test-document-id-C"
    );
    expect(retrievedDocuments[2].fragments[0].metadata).toEqual({
      source: "test-source-C",
      retrievalScore: 0.6,
    });
    expect(retrievedDocuments[2].documentId).toBe("test-document-id-C");
    expect(retrievedDocuments[2].metadata).toEqual({
      test: "test metadata for document C",
    });
    expect(retrievedDocuments[2].attributes).toEqual({ type: "webpage" });
  });
});

describe("retrieveDocuments handles proper access control", () => {
  test("returns no documents if none can be accessed", async () => {
    const metadataDB = new InMemoryDocumentMetadataDB({
      "test-document-id-A": {
        ...DOCUMENT_A_METADATA,
        accessPolicies: [alwaysDenyPolicy],
      },
      "test-document-id-B": {
        ...DOCUMENT_B_METADATA,
        accessPolicies: [alwaysDenyPolicy],
      },
      "test-document-id-C": {
        ...DOCUMENT_C_METADATA,
        accessPolicies: [alwaysDenyPolicy],
      },
    });

    const vectorDB = new TestVectorDB(metadataDB);
    const retriever = new VectorDBDocumentRetriever({ vectorDB, metadataDB });

    const query: VectorDBTextQuery = {
      text: "test",
      topK: 2,
    };

    const retrievedDocuments = await retriever.retrieveData({
      accessPassport: new AccessPassport(),
      query,
    });

    expect(mockQuery).toHaveBeenCalledWith({ topK: 20, text: "test" });
    expect(retrievedDocuments.length).toBe(0);
  });

  test("returns the requested number of document fragments after access filtering", async () => {
    const metadataDB = new InMemoryDocumentMetadataDB({
      "test-document-id-A": {
        ...DOCUMENT_A_METADATA,
        accessPolicies: [alwaysDenyPolicy],
      },
      "test-document-id-B": {
        ...DOCUMENT_B_METADATA,
        accessPolicies: [new AlwaysAllowAccessPolicy()],
      },
      "test-document-id-C": {
        ...DOCUMENT_C_METADATA,
        accessPolicies: [
          /* no policies === can access */
        ],
      },
    });

    const vectorDB = new TestVectorDB(metadataDB);
    const retriever = new VectorDBDocumentRetriever({ vectorDB, metadataDB });

    const query: VectorDBTextQuery = {
      text: "test",
      topK: 2,
    };

    const retrievedDocuments = await retriever.retrieveData({
      accessPassport: new AccessPassport(),
      query,
    });

    expect(mockQuery).toHaveBeenCalledWith({ topK: 20, text: "test" });
    expect(retrievedDocuments.length).toBe(2);

    expect(retrievedDocuments[0].fragments[0].fragmentId).toBe(
      "test-fragment-id-Ba"
    );
    expect(retrievedDocuments[0].fragments[0].documentId).toBe(
      "test-document-id-B"
    );
    expect(retrievedDocuments[0].documentId).toBe("test-document-id-B");

    expect(retrievedDocuments[1].fragments[0].fragmentId).toBe(
      "test-fragment-id-Ca"
    );
    expect(retrievedDocuments[1].fragments[0].documentId).toBe(
      "test-document-id-C"
    );
    expect(retrievedDocuments[1].documentId).toBe("test-document-id-C");
  });

  test("retrieves document fragments and constructs the document with associated fragments", async () => {
    const metadataDB = new InMemoryDocumentMetadataDB({
      "test-document-id-A": {
        ...DOCUMENT_A_METADATA,
        accessPolicies: [new AlwaysAllowAccessPolicy()],
      },
      "test-document-id-B": {
        ...DOCUMENT_B_METADATA,
        accessPolicies: [alwaysDenyPolicy],
      },
      "test-document-id-C": {
        ...DOCUMENT_C_METADATA,
        accessPolicies: [
          /* no policies === can access */
        ],
      },
    });

    const vectorDB = new TestVectorDB(metadataDB);
    const retriever = new VectorDBDocumentRetriever({ vectorDB, metadataDB });

    const query: VectorDBTextQuery = {
      text: "test",
      topK: 3,
    };

    const retrievedDocuments = await retriever.retrieveData({
      accessPassport: new AccessPassport(),
      query,
    });

    expect(mockQuery).toHaveBeenCalledWith({ topK: 30, text: "test" });
    expect(retrievedDocuments.length).toBe(2);

    expect(retrievedDocuments[0].fragments[0].fragmentId).toBe(
      "test-fragment-id-Aa"
    );
    expect(retrievedDocuments[0].fragments[0].documentId).toBe(
      "test-document-id-A"
    );
    expect(retrievedDocuments[0].fragments[1].fragmentId).toBe(
      "test-fragment-id-Ab"
    );
    expect(retrievedDocuments[0].fragments[1].documentId).toBe(
      "test-document-id-A"
    );
    expect(retrievedDocuments[0].documentId).toBe("test-document-id-A");

    expect(retrievedDocuments[1].fragments[0].fragmentId).toBe(
      "test-fragment-id-Ca"
    );
    expect(retrievedDocuments[1].fragments[0].documentId).toBe(
      "test-document-id-C"
    );
    expect(retrievedDocuments[1].documentId).toBe("test-document-id-C");
  });
});
