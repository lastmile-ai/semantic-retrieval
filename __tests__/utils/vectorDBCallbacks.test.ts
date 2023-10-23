import { PineconeVectorDB } from "../../src/data-store/vector-DBs/pineconeVectorDB";
import { InMemoryDocumentMetadataDB } from "../../src/document/metadata/inMemoryDocumentMetadataDB";

// TODO: Should probably move the callback handler tests for pinecone into separate file
import { Pinecone } from "@pinecone-database/pinecone";
import { DirectDocumentParser } from "../../src/ingestion/document-parsers/directDocumentParser";
import { CallbackManager, CallbackMapping } from "../../src/utils/callbacks";
import { TestEmbeddings } from "../__mocks__/transformation/embeddings/testEmbeddings";
import { getTestRawDocument } from "../__utils__/testDocumentUtils";
import TestVectorDB from "../__mocks__/retrieval/testVectorDB";

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

    const testVectorDB = new TestVectorDB(documentMetadataDB);
    testVectorDB.callbackManager = callbackManager;

    try {
      const document = await documentParser.parse(getTestRawDocument());
      await testVectorDB.addDocuments([document]);
      await testVectorDB.query({ topK: 10 });
    } catch (error) {}

    expect(onAddDocumentToVectorDBCallback).toHaveBeenCalled();
    expect(onQueryVectorDBCallback).toHaveBeenCalled();
  });
});
