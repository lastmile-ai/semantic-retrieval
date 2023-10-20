// TODO: These imports should be from actual lastmile retrieval package
import { FileSystem } from "../../src/ingestion/data-sources/fs/fileSystem";

// the import from src/utils/callbacks.ts
import {
  CallbackManager,
  CallbackMapping,
  LoadDocumentsSuccessEvent,
} from "../../src/utils/callbacks";

import type { RawDocument } from "../../src/document/document";
import {
  GoogleDrive,
  OneDrive,
} from "../../src/ingestion/data-sources/dataSource";

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
    const _ = await fileSystem.loadDocuments();

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

    const _ = await fileSystem.loadDocuments();

    // TODO: check that the expected side effect has actually happened.
    for (const onLoadCallback of onLoadSuccessCallbacks) {
      expect(onLoadCallback).toHaveBeenCalled();
    }
    expect(onLoadDocumentsErrorCallback1).not.toHaveBeenCalled();
  });
  // Duplicate the above test for testConnection
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

    const _ = await fileSystem.testConnection();

    // Duplicate the expect calls
    for (const onTestConnectionCallback of onTestConnectionSuccessCallbacks) {
      expect(onTestConnectionCallback).toHaveBeenCalled();
    }
    expect(onTestConnectionErrorCallback1).not.toHaveBeenCalled();
  });

  test("Correct callback called on test connection call, GoogleDrive", async () => {
    const onTestConnectionSuccessCallbacks = [jest.fn(), jest.fn()];
    const onTestConnectionErrorCallback1 = jest.fn();

    const callbacks: CallbackMapping = {
      onDataSourceTestConnectionSuccess: onTestConnectionSuccessCallbacks,
      onDataSourceTestConnectionError: [onTestConnectionErrorCallback1],
    };
    const callbackManager = new CallbackManager("rag-run-0", callbacks);
    const gdrive = new GoogleDrive(callbackManager);

    try {
      const _ = await gdrive.testConnection();
    } catch (error) {
      //expected
    }

    // Duplicate the expect calls
    for (const onTestConnectionCallback of onTestConnectionSuccessCallbacks) {
      expect(onTestConnectionCallback).not.toHaveBeenCalled();
    }
    expect(onTestConnectionErrorCallback1).toHaveBeenCalled();
  });

  test("Correct callback called on test connection call, OneDrive", async () => {
    const onTestConnectionSuccessCallbacks = [jest.fn(), jest.fn()];
    const onTestConnectionErrorCallback1 = jest.fn();

    const callbacks: CallbackMapping = {
      onDataSourceTestConnectionSuccess: onTestConnectionSuccessCallbacks,
      onDataSourceTestConnectionError: [onTestConnectionErrorCallback1],
    };
    const callbackManager = new CallbackManager("rag-run-0", callbacks);
    const gdrive = new OneDrive(callbackManager);

    try {
      const _ = await gdrive.testConnection();
    } catch (error) {
      //expected
    }

    // Duplicate the expect calls
    for (const onTestConnectionCallback of onTestConnectionSuccessCallbacks) {
      expect(onTestConnectionCallback).not.toHaveBeenCalled();
    }
    expect(onTestConnectionErrorCallback1).toHaveBeenCalled();
  });
});
