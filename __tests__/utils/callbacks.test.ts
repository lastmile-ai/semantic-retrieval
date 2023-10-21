// TODO: These imports should be from actual lastmile retrieval package
import { FileSystem } from "../../src/ingestion/data-sources/fs/fileSystem";

import {
  CallbackManager,
  CallbackMapping,
  LoadDocumentsSuccessEvent,
} from "../../src/utils/callbacks";

import type { RawDocument } from "../../src/document/document";

import { DirectDocumentParser } from "../../src/ingestion/document-parsers/directDocumentParser";
import { TextDocumentParser } from "../../src/ingestion/document-parsers/textDocumentParser";
import { getTestRawDocument } from "./testDocumentUtils";

describe("Callbacks", () => {
  test("Callback arg static type", async () => {
    async function onLoadDocumentsSuccessCallback1(
      event: LoadDocumentsSuccessEvent,
      runId: string
    ) {
      const value: RawDocument[] = event.rawDocuments;
      console.log(
        `[runId=${runId}]` +
          "load documents success:\n" +
          JSON.stringify(event.rawDocuments)
      );
    }
    const callbacks: CallbackMapping = {
      onLoadDocumentsSuccess: [onLoadDocumentsSuccessCallback1],
    };
    const callbackManager = new CallbackManager("rag-run-0", callbacks);
    const fileSystem = new FileSystem(
      "./examples/example_data/DonQuixote.txt",
      undefined,
      undefined,
      callbackManager
    );
    await fileSystem.loadDocuments();

    // This test passes by virtue of static type checking. No dynamic condition to check.
    expect(1).toBe(1);
  });
  test("Correct callback called on load docs call, FileSystem", async () => {
    const onLoadSuccessCallbacks = [jest.fn(), jest.fn()];
    const onLoadDocumentsErrorCallback1 = jest.fn();

    const callbacks: CallbackMapping = {
      onLoadDocumentsSuccess: onLoadSuccessCallbacks,
      onLoadDocumentsError: [onLoadDocumentsErrorCallback1],
    };
    const callbackManager = new CallbackManager("rag-run-0", callbacks);
    const fileSystem = new FileSystem(
      "./examples/example_data/DonQuixote.txt",
      undefined,
      undefined,
      callbackManager
    );

    await fileSystem.loadDocuments();

    // TODO: check that the expected side effect has actually happened.
    for (const onLoadCallback of onLoadSuccessCallbacks) {
      expect(onLoadCallback).toHaveBeenCalled();
    }
    expect(onLoadDocumentsErrorCallback1).not.toHaveBeenCalled();
  });

  test("Correct callback called on test connection call, FileSystem", async () => {
    const onTestConnectionSuccessCallbacks = [jest.fn(), jest.fn()];
    const onTestConnectionErrorCallback1 = jest.fn();

    const callbacks: CallbackMapping = {
      onDataSourceTestConnectionSuccess: onTestConnectionSuccessCallbacks,
      onDataSourceTestConnectionError: [onTestConnectionErrorCallback1],
    };
    const callbackManager = new CallbackManager("rag-run-0", callbacks);
    const fileSystem = new FileSystem(
      "./examples/example_data/DonQuixote.txt",
      undefined,
      undefined,
      callbackManager
    );

    await fileSystem.testConnection();

    for (const onTestConnectionCallback of onTestConnectionSuccessCallbacks) {
      expect(onTestConnectionCallback).toHaveBeenCalled();
    }
    expect(onTestConnectionErrorCallback1).not.toHaveBeenCalled();
  });

  test("Direct Document Parser", async () => {
    const onParseNextErrorCallbacks = [jest.fn(), jest.fn()];
    const onParseErrorCallback1 = jest.fn();
    const onParseSuccessCallback1 = jest.fn();

    const callbacks: CallbackMapping = {
      onParseNextError: onParseNextErrorCallbacks,
      onParseError: [onParseErrorCallback1],
      onParseSuccess: [onParseSuccessCallback1],
    };
    const callbackManager = new CallbackManager("rag-run-0", callbacks);
    const documentParser = new DirectDocumentParser(
      undefined,
      undefined,
      callbackManager
    );

    try {
      await documentParser.parse(getTestRawDocument());
    } catch (error) {}
    expect(onParseSuccessCallback1).toHaveBeenCalled();

    expect(onParseErrorCallback1).not.toHaveBeenCalled();
  });
});