import { JSONObject } from "../../src/common/jsonTypes";
import {
  Document,
  DocumentFragment,
  DocumentFragmentType,
  RawDocument,
  RawDocumentChunk,
} from "../../src/document/document";
import { DataSource } from "../../src/ingestion/data-sources/dataSource";
import { FileSystem } from "../../src/ingestion/data-sources/fs/fileSystem";
import { v4 as uuid } from "uuid";

export const TEST_FRAGMENT_TEXT = `
Lorem Ipsum is simply dummy text of the printing and typesetting industry. 
Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown 
printer took a galley of type and scrambled it to make a type specimen book. It has survived 
not only five centuries, but also the leap into electronic typesetting, remaining essentially 
unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum 
passages, and more recently with desktop publishing software like Aldus PageMaker including versions 
of Lorem Ipsum`;

export type TestDocumentConfig = {
  metadata?: JSONObject;
  attributes?: JSONObject;
  documentId?: string;
  rawDocument?: RawDocument;
  fragments?: DocumentFragment[];
};

export function getTestDocument(config?: TestDocumentConfig): Document {
  const documentId = config?.documentId ?? uuid();
  return {
    metadata: config?.metadata ?? {},
    attributes: config?.attributes ?? {},
    documentId,
    rawDocument: config?.rawDocument ?? getTestRawDocument(),
    serialize: async () => "test_path",
    fragments: config?.fragments ?? [
      {
        fragmentId: uuid(),
        fragmentType: "text",
        documentId,
        metadata: {},
        attributes: {},
        getContent: async () => TEST_FRAGMENT_TEXT,
        serialize: async () => TEST_FRAGMENT_TEXT,
      },
    ],
  };
}

export type TestDocumentFragmentConfig = {
  metadata?: JSONObject;
  attributes?: JSONObject;
  fragmentId?: string;
  documentId?: string;
  fragmentType?: DocumentFragmentType;
  content?: string;
  serializedContent?: string;
};

export function getTestDocumentFragment(
  config?: TestDocumentFragmentConfig
): DocumentFragment {
  const fragmentId = config?.fragmentId ?? uuid();
  const documentId = config?.documentId ?? uuid();
  return {
    fragmentId,
    fragmentType: config?.fragmentType ?? "text",
    documentId,
    metadata: config?.metadata ?? {},
    attributes: config?.attributes ?? {},
    getContent: async () => config?.content ?? TEST_FRAGMENT_TEXT,
    serialize: async () => config?.serializedContent ?? TEST_FRAGMENT_TEXT,
  };
}

export type TestRawDocumentConfig = {
  metadata?: JSONObject;
  attributes?: JSONObject;
  documentId?: string;
  dataSource: DataSource;
  chunkedContent?: RawDocumentChunk[];
};

export function getTestRawDocument(
  config?: TestRawDocumentConfig
): RawDocument {
  const documentId = config?.documentId ?? uuid();
  return {
    metadata: config?.metadata ?? {},
    attributes: config?.attributes ?? {},
    documentId,
    uri: "test-source-uri",
    name: "test-raw-doc-name",
    dataSource: config?.dataSource ?? new FileSystem("test"),
    mimeType: "text/plain",
    getChunkedContent: async () =>
      config?.chunkedContent ?? [
        {
          content: TEST_FRAGMENT_TEXT,
          metadata: { source: "test-source-uri" },
        },
      ],
    getContent: async () => TEST_FRAGMENT_TEXT,
  };
}
