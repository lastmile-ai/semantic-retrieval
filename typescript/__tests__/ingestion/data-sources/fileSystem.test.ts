// TODO(LAS-399): Figure out proper project build structure to prevent lint error here
import { RawDocument } from "../../../src/document/document";
import { FileSystem } from "../../../src/ingestion/data-sources/fs/fileSystem";
import * as path from "node:path";

// TODO(LAS-400): NODE_OPTIONS='--experimental-vm-modules' is required for jest command to support dynamic import used
// by langchain. We need to handle dynamic imports in a better way (e.g. a custom build/transformation step)

function resolveExamplesPath(examplePath: string = "") {
  return path.resolve(
    `../examples/example_data/ingestion${examplePath ? "/" : ""}${examplePath}`
  );
}

function trimWhitespace(text: string) {
  return text.replace(/(\r\n|\n|\r)/gm, " ");
}

function validateRawDocumentMetadata(
  rawDocument: RawDocument,
  path: string,
  mimeType: string
) {
  expect(rawDocument.uri).toBe(path);
  expect(rawDocument.name).toBe(path);
  expect(rawDocument.dataSource.name).toBe("FileSystem");
  expect(rawDocument.mimeType).toBe(mimeType);
}

async function validateTxtRawDocument(rawDocument: RawDocument) {
  const path = resolveExamplesPath("DonQuixote.txt");
  validateRawDocumentMetadata(rawDocument, path, "text/plain");

  const chunkedContent = await rawDocument.getChunkedContent();
  expect(chunkedContent.length).toBe(1);
  expect(chunkedContent[0].content).toContain("In a village of La Mancha");
  expect(chunkedContent[0].metadata.source).toBe(path);
}

async function validatePdfRawDocument(rawDocument: RawDocument) {
  const path = resolveExamplesPath("Introduction_to_AI_Chapter1.pdf");
  validateRawDocumentMetadata(rawDocument, path, "application/pdf");
  const chunkedContent = await rawDocument.getChunkedContent();
  expect(chunkedContent.length).toBe(14); // 14 pages
  const firstPage = trimWhitespace(chunkedContent[0].content);
  expect(firstPage).toContain(
    "The term artificial intelligence stirs emotions"
  );
  expect(firstPage).toContain(
    "The speed of each motor is influenced by a light sensor on"
  );
  expect(firstPage).not.toContain("the front of the vehicle");
  expect(chunkedContent[0].metadata.source).toBe(path);
  expect(
    (chunkedContent[0].metadata.loc as { pageNumber: number }).pageNumber
  ).toBe(1);
  expect(trimWhitespace(chunkedContent[1].content)).toContain(
    "the front of the vehicle"
  );
  expect(chunkedContent[1].metadata.source).toBe(path);
}

async function validateCsvRawDocument(rawDocument: RawDocument) {
  const path = resolveExamplesPath("mlb_teams_2012.csv");
  validateRawDocumentMetadata(rawDocument, path, "text/csv");
}

async function validateDocxRawDocument(rawDocument: RawDocument) {
  const path = resolveExamplesPath("Role_of_AI_in_SE_and_Testing.docx");
  validateRawDocumentMetadata(
    rawDocument,
    path,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
  );
  // TODO: current segfaulting when getting chunked content for csv and docx... Need to fix
  // const chunkedContent = await rawDocument.getChunkedContent();
  // expect(chunkedContent.length).toBe(3); // 14 pages
}

describe("FileSystem data source", () => {
  test("individual txt file", async () => {
    const path = resolveExamplesPath("DonQuixote.txt");
    const fileSystem = new FileSystem({ path });
    const rawDocuments = await fileSystem.loadDocuments();

    expect(rawDocuments.length).toBe(1);
    await validateTxtRawDocument(rawDocuments[0]);
  });

  test("directory", async () => {
    const directoryPath = resolveExamplesPath();
    const rawDocuments = await new FileSystem({
      path: directoryPath,
    }).loadDocuments();
    expect(rawDocuments.length).toBe(4);

    await validateTxtRawDocument(rawDocuments[0]);
    await validatePdfRawDocument(rawDocuments[1]);
    await validateDocxRawDocument(rawDocuments[2]);
    await validateCsvRawDocument(rawDocuments[3]);
  });
});
