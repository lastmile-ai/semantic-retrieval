import { DocumentMetadata } from "../../src/document/metadata/documentMetadata";
import { InMemoryDocumentMetadataDB } from "../../src/document/metadata/inMemoryDocumentMetadataDB";
import {
  getTestDocument,
  getTestRawDocument,
} from "../__utils__/testDocumentUtils";
import { AlwaysAllowAccessPolicy } from "../../src/access-control/policies/alwaysAllowAccessPolicy";

const TEST_DOCUMENT_ID_1 = "test-document-id-1";
const TEST_DOCUMENT_ID_2 = "test-document-id-2";

const TEST_METADATA_1: DocumentMetadata = {
  documentId: TEST_DOCUMENT_ID_1,
  rawDocument: getTestRawDocument({ documentId: TEST_DOCUMENT_ID_1 }),
  document: getTestDocument({ documentId: TEST_DOCUMENT_ID_1 }),
  collectionId: "test-collection-id",
  uri: "test-document-1-uri",
  accessPolicies: [new AlwaysAllowAccessPolicy()],
  metadata: {
    key1: "value1",
    key2: "value2",
  },
};

const TEST_METADATA_2: DocumentMetadata = {
  documentId: TEST_DOCUMENT_ID_2,
  rawDocument: getTestRawDocument({ documentId: TEST_DOCUMENT_ID_2 }),
  document: getTestDocument({ documentId: TEST_DOCUMENT_ID_2 }),
  uri: "test-document-2-uri",
  metadata: {
    key1: "value A",
    key2: "value2",
    key3: "Look for random value here",
  },
};

const QUERY_METADATA_DB = new InMemoryDocumentMetadataDB({
  [TEST_DOCUMENT_ID_1]: TEST_METADATA_1,
  [TEST_DOCUMENT_ID_2]: TEST_METADATA_2,
});

describe("inMemoryDocumentMetadataDB", () => {
  test("get metadata", async () => {
    const metadataDB = new InMemoryDocumentMetadataDB({
      [TEST_DOCUMENT_ID_1]: TEST_METADATA_1,
    });
    expect(await metadataDB.getMetadata(TEST_DOCUMENT_ID_1)).toBe(
      TEST_METADATA_1
    );
  });

  test("set metadata", async () => {
    const metadataDB = new InMemoryDocumentMetadataDB();
    await metadataDB.setMetadata(TEST_DOCUMENT_ID_1, TEST_METADATA_1);
    expect(await metadataDB.getMetadata(TEST_DOCUMENT_ID_1)).toBe(
      TEST_METADATA_1
    );
  });

  test("set metadata overrides existing metadata", async () => {
    const metadataDB = new InMemoryDocumentMetadataDB({
      [TEST_DOCUMENT_ID_1]: TEST_METADATA_1,
    });

    await metadataDB.setMetadata(TEST_DOCUMENT_ID_1, {
      ...TEST_METADATA_1,
      collectionId: "new-collection-id",
    });

    expect(await metadataDB.getMetadata(TEST_DOCUMENT_ID_1)).toEqual({
      ...TEST_METADATA_1,
      collectionId: "new-collection-id",
    });
  });

  test("persist to, and load from JSON file", async () => {
    const metadataDB = new InMemoryDocumentMetadataDB({
      [TEST_DOCUMENT_ID_1]: TEST_METADATA_1,
    });

    await metadataDB.persist(
      "__tests__/__mocks__/document/testDocumentMetadataDB.json"
    );

    const metadataDBFromJSON = await InMemoryDocumentMetadataDB.fromJSONFile(
      "__tests__/__mocks__/document/testDocumentMetadataDB.json"
    );

    const retrievedMetadata =
      await metadataDBFromJSON.getMetadata(TEST_DOCUMENT_ID_1);

    expect(retrievedMetadata.documentId).toBe(TEST_DOCUMENT_ID_1);
    expect(retrievedMetadata.collectionId).toBe(TEST_METADATA_1.collectionId);
    expect(retrievedMetadata.uri).toBe(TEST_METADATA_1.uri);
    expect(retrievedMetadata.rawDocument?.uri).toBe("test-source-uri");
    expect(retrievedMetadata.accessPolicies).toEqual(
      TEST_METADATA_1.accessPolicies
    );
  });

  test("query document ids from metadata", async () => {
    expect(
      await QUERY_METADATA_DB.queryDocumentIds({
        type: "metadata",
        metadataKey: "key2",
        metadataValue: "value2",
        matchType: "exact",
      })
    ).toEqual([TEST_DOCUMENT_ID_1, TEST_DOCUMENT_ID_2]);
    expect(
      await QUERY_METADATA_DB.queryDocumentIds({
        type: "metadata",
        metadataKey: "key1",
        metadataValue: "value A",
        matchType: "exact",
      })
    ).toEqual([TEST_DOCUMENT_ID_2]);
    expect(
      await QUERY_METADATA_DB.queryDocumentIds({
        type: "metadata",
        metadataKey: "key3",
        metadataValue: "random",
        matchType: "includes",
      })
    ).toEqual([TEST_DOCUMENT_ID_2]);
    expect(
      await QUERY_METADATA_DB.queryDocumentIds({
        type: "metadata",
        metadataKey: "key3",
        metadataValue: "random",
        matchType: "exact",
      })
    ).toEqual([]);
  });

  test("query document ids from string fields", async () => {
    expect(
      await QUERY_METADATA_DB.queryDocumentIds({
        type: "string_field",
        fieldName: "uri",
        fieldValue: "test-document-2-uri",
        matchType: "exact",
      })
    ).toEqual([TEST_DOCUMENT_ID_2]);
    expect(
      await QUERY_METADATA_DB.queryDocumentIds({
        type: "string_field",
        fieldName: "uri",
        fieldValue: "test-document",
        matchType: "includes",
      })
    ).toEqual([TEST_DOCUMENT_ID_1, TEST_DOCUMENT_ID_2]);
    expect(
      await QUERY_METADATA_DB.queryDocumentIds({
        type: "string_field",
        fieldName: "collectionId",
        fieldValue: "test",
        matchType: "includes",
      })
    ).toEqual([TEST_DOCUMENT_ID_1]);
    expect(
      await QUERY_METADATA_DB.queryDocumentIds({
        type: "string_field",
        fieldName: "uri",
        fieldValue: "test-dne",
        matchType: "includes",
      })
    ).toEqual([]);
  });
});
