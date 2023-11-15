// TODO(LAS-399): Figure out proper project build structure to prevent lint error here
import { RawDocument } from "../../../src/document/document";
import { FileSystem } from "../../../src/ingestion/data-sources/fs/fileSystem";
import { DirectDocumentParser } from "../../../src/ingestion/document-parsers/directDocumentParser";

const testRawContent1 = `
Lorem Ipsum is simply dummy text of the printing and typesetting industry. 
Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown 
printer took a galley of type and scrambled it to make a type specimen book. It has survived 
not only five centuries, but also the leap into electronic typesetting, remaining essentially 
unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum 
passages, and more recently with desktop publishing software like Aldus PageMaker including versions 
of Lorem Ipsum`;

const testRawContent2 = `
Contrary to popular belief, Lorem Ipsum is not simply random text. It has roots in a piece of 
classical Latin literature from 45 BC, making it over 2000 years old. Richard McClintock, a Latin 
professor at Hampden-Sydney College in Virginia, looked up one of the more obscure Latin words, 
consectetur, from a Lorem Ipsum passage, and going through the cites of the word in classical literature, 
discovered the undoubtable source. 
`;

const testRawContent3 = `
Lorem Ipsum comes from sections 1.10.32 and 1.10.33 of 
"de Finibus Bonorum et Malorum" (The Extremes of Good and Evil) by Cicero, written in 45 BC. This book 
is a treatise on the theory of ethics, very popular during the Renaissance. The first line of Lorem Ipsum, 
"Lorem ipsum dolor sit amet..", comes from a line in section 1.10.32.
`;

const rawDocument1: RawDocument = {
  metadata: {},
  attributes: {},
  documentId: "rawDoc1Id",
  uri: "rawDoc1Uri",
  name: "rawDoc1Name",
  dataSource: new FileSystem({ path: "test" }),
  mimeType: "text/plain",
  getChunkedContent: async () => [
    {
      content: testRawContent1,
      metadata: { source: "rawDoc1Uri" },
    },
  ],
  getContent: async () => testRawContent1,
};

const rawDocument2: RawDocument = {
  metadata: {},
  attributes: {},
  documentId: "rawDoc2Id",
  uri: "rawDoc2Uri",
  name: "rawDoc2Name",
  dataSource: new FileSystem({ path: "test" }),
  mimeType: "text/plain",
  getChunkedContent: async () => [
    {
      content: testRawContent1,
      metadata: { source: "rawDoc2Uri" },
    },
    {
      content: testRawContent2,
      metadata: { source: "rawDoc2Uri" },
    },
    {
      content: testRawContent3,
      metadata: { source: "rawDoc2Uri" },
    },
  ],
  getContent: async () => testRawContent2,
};

describe("DirectDocument parser", () => {
  test("parse single RawDocumentChunk into matching DocumentFragment", async () => {
    const document = await new DirectDocumentParser().parse(rawDocument1);
    expect(document.rawDocument).toBe(rawDocument1);
    expect(document.documentId).not.toBe(rawDocument1.documentId);
    const fragments = document.fragments;
    expect(fragments.length).toBe(1);
    expect(await fragments[0].getContent()).toBe(testRawContent1);
  });

  test("parse multiple RawDocumentChunks into matching DocumentFragments", async () => {
    const document = await new DirectDocumentParser().parse(rawDocument2);
    expect(document.rawDocument).toBe(rawDocument2);
    const fragments = document.fragments;
    expect(fragments.length).toBe(3);
    expect(await fragments[0].getContent()).toBe(testRawContent1);
    expect(await fragments[1].getContent()).toBe(testRawContent2);
    expect(await fragments[2].getContent()).toBe(testRawContent3);
  });
});
