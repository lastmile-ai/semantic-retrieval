import { DocumentMetadata } from "../../src/document/metadata/documentMetadata";
import { InMemoryDocumentMetadataDB } from "../../src/document/metadata/inMemoryDocumentMetadataDB";
import {
  getTestDocument,
  getTestRawDocument,
} from "../__utils__/testDocumentUtils";
import { AlwaysAllowAccessPolicy } from "../../src/access-control/policies/alwaysAllowAccessPolicy";

const TEST_DOCUMENT_ID = "test-document-id";
const TEST_METADATA: DocumentMetadata = {
  documentId: TEST_DOCUMENT_ID,
  rawDocument: getTestRawDocument({ documentId: TEST_DOCUMENT_ID }),
  document: getTestDocument({ documentId: TEST_DOCUMENT_ID }),
  collectionId: "test-collection-id",
  uri: "test-document-uri",
  accessPolicies: [new AlwaysAllowAccessPolicy()],
};

describe("inMemoryDocumentMetadataDB", () => {
  test("get metadata", async () => {
    const metadataDB = new InMemoryDocumentMetadataDB({
      [TEST_DOCUMENT_ID]: TEST_METADATA,
    });
    expect(await metadataDB.getMetadata(TEST_DOCUMENT_ID)).toBe(TEST_METADATA);
  });

  test("set metadata", async () => {
    const metadataDB = new InMemoryDocumentMetadataDB();
    await metadataDB.setMetadata(TEST_DOCUMENT_ID, TEST_METADATA);
    expect(await metadataDB.getMetadata(TEST_DOCUMENT_ID)).toBe(TEST_METADATA);
  });

  test("set metadata overrides existing metadata", async () => {
    const metadataDB = new InMemoryDocumentMetadataDB({
      [TEST_DOCUMENT_ID]: TEST_METADATA,
    });

    await metadataDB.setMetadata(TEST_DOCUMENT_ID, {
      ...TEST_METADATA,
      collectionId: "new-collection-id",
    });

    expect(await metadataDB.getMetadata(TEST_DOCUMENT_ID)).toEqual({
      ...TEST_METADATA,
      collectionId: "new-collection-id",
    });
  });

  test("persist to, and load from JSON file", async () => {
    const metadataDB = new InMemoryDocumentMetadataDB({
      [TEST_DOCUMENT_ID]: TEST_METADATA,
    });

    await metadataDB.persist(
      "__tests__/__mocks__/document/testDocumentMetadataDB.json"
    );

    const metadataDBFromJSON = await InMemoryDocumentMetadataDB.fromJSONFile(
      "__tests__/__mocks__/document/testDocumentMetadataDB.json"
    );

    const retrievedMetadata =
      await metadataDBFromJSON.getMetadata(TEST_DOCUMENT_ID);

    expect(retrievedMetadata.documentId).toBe(TEST_DOCUMENT_ID);
    expect(retrievedMetadata.collectionId).toBe(TEST_METADATA.collectionId);
    expect(retrievedMetadata.uri).toBe(TEST_METADATA.uri);
    expect(retrievedMetadata.rawDocument?.uri).toBe("test-source-uri");
    expect(retrievedMetadata.accessPolicies).toEqual(
      TEST_METADATA.accessPolicies
    );
  });
});
