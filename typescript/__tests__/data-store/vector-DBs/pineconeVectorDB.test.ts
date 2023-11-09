import { Pinecone } from "@pinecone-database/pinecone";
import getEnvVar from "../../../src/utils/getEnvVar";
import { PineconeVectorDB } from "../../../src/data-store/vector-DBs/pineconeVectorDB";
import { InMemoryDocumentMetadataDB } from "../../../src/document/metadata/inMemoryDocumentMetadataDB";
import {
  TEST_VECTOR,
  TestEmbeddings,
} from "../../__mocks__/transformation/embeddings/testEmbeddings";
import {
  getTestDocument,
  getTestDocumentFragment,
  getTestRawDocument,
} from "../../__utils__/testDocumentUtils";
import { VectorDBTextQuery } from "../../../src/data-store/vector-DBs/vectorDB";
import { CallbackManager, CallbackMapping } from "../../../src/utils/callbacks";
import { DirectDocumentParser } from "../../../src/ingestion/document-parsers/directDocumentParser";

jest.mock("../../../src/utils/getEnvVar");
jest.mock("uuid");

function mockPineconeDescribeIndexStats(): Promise<{
  dimensions: number;
  numRecords: number;
}> {
  return Promise.resolve({ dimensions: 1536, numRecords: 0 });
}

const mockUpsert = jest.fn();
const mockQuery = jest.fn();
const mockDescribeIndexStats = jest
  .fn()
  .mockImplementation(mockPineconeDescribeIndexStats);

jest.mock("uuid", () => ({
  ...jest.requireActual("uuid"),
  v4: jest.fn().mockReturnValue("test-uuid"),
}));

jest.mock("@pinecone-database/pinecone", () => {
  const originalModule = jest.requireActual("@pinecone-database/pinecone");

  const index = jest.fn().mockImplementation(() => ({
    namespace: jest.fn().mockImplementation(() => ({
      query: mockQuery,
      upsert: mockUpsert,
    })),
    describeIndexStats: mockDescribeIndexStats,
  }));

  return {
    ...originalModule,
    Index: index,
    Pinecone: jest.fn().mockImplementation(() => ({
      index,
    })),
  };
});

const mockedPinecone = Pinecone as jest.MockedClass<typeof Pinecone>;
const mockedGetEnvVar = getEnvVar as jest.MockedFunction<typeof getEnvVar>;

const TEST_CONFIG = {
  indexName: "test-index",
  embeddings: new TestEmbeddings(),
  metadataDB: new InMemoryDocumentMetadataDB(),
};

describe("Pinecone API key and environment validation", () => {
  beforeEach(() => {
    mockedGetEnvVar.mockClear();
    mockedGetEnvVar.mockImplementation((key) => {
      if (key === "PINECONE_API_KEY") {
        return "test-env-api-key";
      } else if (key === "PINECONE_ENVIRONMENT") {
        return "test-env-environment";
      }
    });
  });

  test("error thrown if no keys are available", () => {
    mockedGetEnvVar.mockReturnValue(undefined);
    expect(() => new PineconeVectorDB(TEST_CONFIG)).toThrowError(
      "No Pinecone API key found for PineconeVectorDB"
    );

    expect(
      () => new PineconeVectorDB({ ...TEST_CONFIG, apiKey: "test-config-key" })
    ).toThrowError("No Pinecone environment found for PineconeVectorDB");
  });

  test("uses keys from config when provided", () => {
    expect(
      () =>
        new PineconeVectorDB({
          ...TEST_CONFIG,
          apiKey: "test-config-key",
          environment: "test-config-environment",
        })
    ).not.toThrow();
    expect(mockedPinecone).toHaveBeenCalledWith({
      apiKey: "test-config-key",
      environment: "test-config-environment",
    });
  });

  test("uses keys from environment when no config keys provided", () => {
    expect(() => new PineconeVectorDB(TEST_CONFIG)).not.toThrow();
    expect(mockedPinecone).toHaveBeenCalledWith({
      apiKey: "test-env-api-key",
      environment: "test-env-environment",
    });
  });
});

describe("pineconeVectorDB addDocuments", () => {
  beforeAll(() => {
    mockedGetEnvVar.mockImplementation((key) => {
      if (key === "PINECONE_API_KEY") {
        return "test-env-api-key";
      } else if (key === "PINECONE_ENVIRONMENT") {
        return "test-env-environment";
      }
    });
  });

  beforeEach(() => {
    mockUpsert.mockClear();
  });

  test("upserts vectors with correct metadata", async () => {
    const fragment = getTestDocumentFragment({
      content: "test-fragment-1",
      metadata: {
        source: "test-source",
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
      },
    });

    const document = getTestDocument({
      fragments: [fragment],
    });

    const documents = [document];

    const vectorDB = new PineconeVectorDB(TEST_CONFIG);
    await vectorDB.addDocuments(documents);
    expect(mockUpsert).toHaveBeenCalledWith([
      {
        values: TEST_VECTOR,
        id: "test-uuid",
        metadata: {
          source: "test-source",
          stringArray: ["this", "is", "fine"],
          num: 1,
          bool: true,
          documentId: document.documentId,
          fragmentId: fragment.fragmentId,
          "nested.nestedString": "nested",
          "nested.nestedNum": 2,
          "nested.nestedBool": false,
          "nested.nestedStringArray": ["all", "good"],
          "nested.doubleNested.val": "test",
          text: "test-fragment-1",
        },
      },
    ]);
  });

  test("upserts many vectors in correct batches", async () => {
    // 3 documents of 400 fragments each will have 12 parallel requests of 100
    // vectors each
    const testDocuments = Array(3).fill(
      getTestDocument({
        fragments: Array(400).fill(getTestDocumentFragment()),
      })
    );

    await PineconeVectorDB.fromDocuments(testDocuments, TEST_CONFIG);
    expect(mockUpsert).toHaveBeenCalledTimes(12);
  });
});

describe("pineconeVectorDB query", () => {
  beforeAll(() => {
    mockedGetEnvVar.mockImplementation((key) => {
      if (key === "PINECONE_API_KEY") {
        return "test-env-api-key";
      } else if (key === "PINECONE_ENVIRONMENT") {
        return "test-env-environment";
      }
    });
  });

  beforeEach(() => {
    mockQuery.mockClear();
  });

  test("returns vectors with proper metadata from results", async () => {
    mockQuery.mockImplementationOnce(async () => ({
      matches: [
        {
          values: TEST_VECTOR,
          id: "test-uuid",
          metadata: {
            source: "test-source",
            stringArray: ["this", "is", "fine"],
            num: 1,
            bool: true,
            documentId: "test-document-id",
            fragmentId: "test-fragment-id",
            "nested.nestedString": "nested",
            "nested.nestedNum": 2,
            "nested.nestedBool": false,
            "nested.nestedStringArray": ["all", "good"],
            "nested.doubleNested.val": "test",
            text: "test-fragment-1",
          },
          score: 0.5,
        },
      ],
      namespace: "",
    }));

    const results = await new PineconeVectorDB(TEST_CONFIG).query({
      text: "test",
      topK: 1,
    } as VectorDBTextQuery);

    expect(results).toEqual([
      {
        vector: TEST_VECTOR,
        text: "test-fragment-1",
        metadata: {
          source: "test-source",
          stringArray: ["this", "is", "fine"],
          num: 1,
          bool: true,
          documentId: "test-document-id",
          fragmentId: "test-fragment-id",
          nested: {
            nestedString: "nested",
            nestedNum: 2,
            nestedBool: false,
            nestedStringArray: ["all", "good"],
            doubleNested: {
              val: "test",
            },
          },
          retrievalScore: 0.5,
        },
        attributes: {},
      },
    ]);
  });

  test("callbacks are called", async () => {
    mockQuery.mockImplementationOnce(async () => {
      return { matches: [] };
    });

    const onAddDocumentToVectorDBCallback = jest.fn();
    const onQueryVectorDBCallback = jest.fn();

    const callbacks: CallbackMapping = {
      onAddDocumentsToVectorDB: [onAddDocumentToVectorDBCallback],
      onQueryVectorDB: [onQueryVectorDBCallback],
    };
    const callbackManager = new CallbackManager("rag-run-0", callbacks);
    const documentParser = new DirectDocumentParser();

    const embeddingsTransformer = new TestEmbeddings();
    const documentMetadataDB = new InMemoryDocumentMetadataDB();

    const pineconeVectorDB = new PineconeVectorDB({
      apiKey: "1",
      environment: "a",
      indexName: "b",
      embeddings: embeddingsTransformer,
      metadataDB: documentMetadataDB,
    });
    pineconeVectorDB.callbackManager = callbackManager;

    try {
      const document = await documentParser.parse(getTestRawDocument());
      await pineconeVectorDB.addDocuments([document]);
      await pineconeVectorDB.query({ topK: 10, text: "test" });
    } catch (error) {}

    expect(onAddDocumentToVectorDBCallback).toHaveBeenCalled();
    expect(onQueryVectorDBCallback).toHaveBeenCalled();
  });
});
