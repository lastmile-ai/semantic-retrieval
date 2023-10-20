// TODO: These imports should be from actual lastmile retrieval package
import { FileSystem } from "../../src/ingestion/data-sources/fs/fileSystem";

// the import from src/utils/callbacks.ts
import {
  CallbackManager,
  CallbackMapping,
  LoadDocumentsSuccessEvent,
  LoadDocumentsErrorEvent,
} from "../../src/utils/callbacks";

import type { RawDocument } from "../../src/document/document";

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
      // onLoadDocumentsError: [onLoadDocumentsErrorCallback1]
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
  test("Correct callback called on event", async () => {
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
});
