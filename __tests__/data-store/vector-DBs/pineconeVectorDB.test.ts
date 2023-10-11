import {
  Pinecone,
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

  return {
    ...originalModule,
    PineCone: jest.fn().mockImplementation(() => ({
      Index: {
        namespace: {
          query: mockQuery,
          upsert: mockUpsert,
        },
        describeIndexStats: mockDescribeIndexStats,
      },
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

describe("Pinecone API key validation", () => {
  test("error thrown if no keys are available", () => {
    expect(() => new PineconeVectorDB(TEST_CONFIG)).toThrowError(
      "No Pinecone API key found for PineconeVectorDB"
    );
  });

  test("uses key from config when provided", () => {
    mockedGetEnvVar.mockReturnValue("test-env-key");
    expect(
      () => new PineconeVectorDB({ ...TEST_CONFIG, apiKey: "test-config-key" })
    ).not.toThrow();
    expect(mockedPinecone).toHaveBeenCalledWith({
      ...TEST_CONFIG,
      apiKey: "test-config-key",
    });
  });

  test("uses key from environment when no config key provided", () => {
    mockedGetEnvVar.mockReturnValue("test-env-key");
    expect(() => new PineconeVectorDB(TEST_CONFIG)).not.toThrow();
    expect(mockedPinecone).toHaveBeenCalledWith({
      ...TEST_CONFIG,
      apiKey: "test-env-key",
    });
  });
});

describe("sanitizeMetadata", () => {});

describe("pineconeVectorDB addDocuments", () => {});

describe("pineconeVectorDB query", () => {});
