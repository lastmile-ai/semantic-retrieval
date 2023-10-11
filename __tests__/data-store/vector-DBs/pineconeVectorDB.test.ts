import {
  Pinecone,
  Index,
  PineconeRecord,
  QueryOptions,
  RecordMetadata,
  ScoredPineconeRecord,
} from "@pinecone-database/pinecone";
import getEnvVar from "../../../src/utils/getEnvVar";
import { PineconeVectorDB } from "../../../src/data-store/vector-DBs/pineconeVectorDB";
import { OpenAIEmbeddings } from "../../../src/transformation/embeddings/openAIEmbedding";
import { InMemoryDocumentMetadataDB } from "../../../src/document/metadata/InMemoryDocumentMetadataDB";
import { TestEmbeddings } from "../../__mocks__/transformation/embeddings/TestEmbeddings";

jest.mock("../../../src/utils/getEnvVar");

function mockPineconeUpsert(
  data: Array<PineconeRecord<RecordMetadata>>
): Promise<void> {
  return Promise.resolve();
}

function mockPineconeQuery(options: QueryOptions): Promise<{
  matches?: Array<ScoredPineconeRecord<RecordMetadata>>;
  namespace: string;
}> {
  return Promise.resolve({ matches: [], namespace: "" });
}

function mockPineconeDescribeIndexStats(): Promise<{
  dimensions: number;
  numRecords: number;
}> {
  return Promise.resolve({ dimensions: 1536, numRecords: 0 });
}

const mockUpsert = jest.fn().mockImplementation(mockPineconeUpsert);
const mockQuery = jest.fn().mockImplementation(mockPineconeQuery);
const mockDescribeIndexStats = jest
  .fn()
  .mockImplementation(mockPineconeDescribeIndexStats);

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

describe("sanitizeMetadata", () => {});

describe("pineconeVectorDB addDocuments", () => {});

describe("pineconeVectorDB query", () => {});
