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
  metadata: {
    key1: "value1",
    key2: "value2",
  },
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

  test("query document ids from metadata", async () => {
    const testDocumentID2 = "test-document-id-2";
    const metadataDB = new InMemoryDocumentMetadataDB({
      [TEST_DOCUMENT_ID]: TEST_METADATA,
      [testDocumentID2]: {
        documentId: testDocumentID2,
        rawDocument: getTestRawDocument({ documentId: testDocumentID2 }),
        document: getTestDocument({ documentId: testDocumentID2 }),
        uri: "test-document-2-uri",
        metadata: {
          key1: "value A",
          key2: "value2",
          key3: "Look for random value here",
        },
      },
    });

    expect(
      await metadataDB.queryDocumentIds({
        metadataKey: "key2",
        metadataValue: "value2",
        matchType: "exact",
      })
    ).toEqual([TEST_DOCUMENT_ID, testDocumentID2]);
    expect(
      await metadataDB.queryDocumentIds({
        metadataKey: "key1",
        metadataValue: "value A",
        matchType: "exact",
      })
    ).toEqual([testDocumentID2]);
    expect(
      await metadataDB.queryDocumentIds({
        metadataKey: "key3",
        metadataValue: "random",
        matchType: "includes",
      })
    ).toEqual([testDocumentID2]);
    expect(
      await metadataDB.queryDocumentIds({
        metadataKey: "key3",
        metadataValue: "random",
        matchType: "exact",
      })
    ).toEqual([]);
  });
});
