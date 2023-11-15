import { InMemoryDocumentMetadataDB } from "../../src/document/metadata/inMemoryDocumentMetadataDB";
import { DirectDocumentParser } from "../../src/ingestion/document-parsers/directDocumentParser";
import { CallbackManager, CallbackMapping } from "../../src/utils/callbacks";
import { getTestRawDocument } from "../__utils__/testDocumentUtils";
import TestVectorDB from "../__mocks__/retrieval/testVectorDB";

describe("VectorDBCallbacks", () => {
  test("Vector DB", async () => {
    const onAddDocumentToVectorDBCallback = jest.fn();
    const onQueryVectorDBCallback = jest.fn();

    const callbacks: CallbackMapping = {
      onAddDocumentsToVectorDB: [onAddDocumentToVectorDBCallback],
      onQueryVectorDB: [onQueryVectorDBCallback],
    };
    const callbackManager = new CallbackManager("rag-run-0", callbacks);
    const documentParser = new DirectDocumentParser();

    const documentMetadataDB = new InMemoryDocumentMetadataDB();

    const testVectorDB = new TestVectorDB(documentMetadataDB);
    testVectorDB.callbackManager = callbackManager;

    try {
      const document = await documentParser.parse(getTestRawDocument());
      await testVectorDB.addDocuments([document]);
      await testVectorDB.query({ topK: 10, text: "test" });
    } catch (error) {}

    expect(onAddDocumentToVectorDBCallback).toHaveBeenCalled();
    expect(onQueryVectorDBCallback).toHaveBeenCalled();
  });
});
