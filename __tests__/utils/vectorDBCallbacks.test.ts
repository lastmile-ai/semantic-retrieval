import { PineconeVectorDB } from "../../src/data-store/vector-DBs/pineconeVectorDB";
import { OpenAIEmbeddings } from "../../src/transformation/embeddings/openAIEmbeddings";
import { InMemoryDocumentMetadataDB } from "../../src/document/metadata/inMemoryDocumentMetadataDB";

// TODO: Should probably move the callback handler tests for pinecone into separate file
import { Pinecone } from "@pinecone-database/pinecone";
import { DirectDocumentParser } from "../../src/ingestion/document-parsers/directDocumentParser";
import { CallbackManager, CallbackMapping } from "../../src/utils/callbacks";
import { TestEmbeddings } from "../__mocks__/transformation/embeddings/testEmbeddings";
import { getTestRawDocument } from "../__utils__/testDocumentUtils";

jest.mock("@pinecone-database/pinecone", () => {
  return {
    Pinecone: jest.fn().mockImplementation(() => {
      return {
        index: jest.fn().mockImplementation(() => {
          return {
            namespace: jest.fn().mockImplementation(() => {
              return {
                upsert: jest.fn(),
                query: jest.fn().mockImplementation(() => {
                  return { matches: [] };
                }),
              };
            }),
            describeIndexStats: jest.fn().mockImplementation(() => {
              return new Promise(() => {});
            }),
          };
        }),
      };
    }),
  };
});
jest.mock("../../src/transformation/embeddings/openAIEmbeddings", () => {
  return {
    OpenAIEmbeddings: jest.fn().mockImplementation(() => {
      return {
        transformDocuments: () => {
          return [];
        },
        embed: (text: string) => {
          return { vector: [1, 2, 3] };
        },
      };
    }),
  };
});

describe("VectorDBCallbacks", () => {
  test("Vector DB", async () => {
    const onAddDocumentToVectorDBCallback = jest.fn();
    const onQueryVectorDBCallback = jest.fn();

    const callbacks: CallbackMapping = {
      onAddDocumentToVectorDB: [onAddDocumentToVectorDBCallback],
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
