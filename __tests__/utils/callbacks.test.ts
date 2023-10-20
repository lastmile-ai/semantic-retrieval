// TODO: These imports should be from actual lastmile retrieval package
import { RawDocument } from "../../src/document/document";
import { FileSystem } from "../../src/ingestion/data-sources/fs/fileSystem";

// the import from src/utils/callbacks.ts
import {
  CallbackManager,
  CallbackMapping,
  ILoadDocumentsSuccessCallback,
  ILoadDocumentsSuccessEvent,
} from "../../src/utils/callbacks";

// import { RawDocument } from "../../src/document/document";

describe("Callbacks are called", () => {
  test("returns the documents with correct metadata and count", async () => {
    async function onLoadDocumentsSuccessCallback1(
      event: ILoadDocumentsSuccessEvent,
      rag_run_id: string
    ) {
      console.log(
        `[rag_run_id=${rag_run_id}]` +
          "load documents success:\n" +
          JSON.stringify(event.data)
      );
    }
    // async function onLoadDocumentsErrorCallback1(event: LoadDocumentsErrorEvent, rag_run_id: string) {
    //   console.log(`[rag_run_id=${rag_run_id}]` + "Error message: " + event.message);
    // }
    // const onLoadDocumentsSuccessCallback1 = jest.fn();
    const onLoadDocumentsErrorCallback1 = jest.fn();

    const loadDocumentsSuccessCallback: ILoadDocumentsSuccessCallback =
      onLoadDocumentsErrorCallback1;

    const callbacks: CallbackMapping = {
      onLoadDocumentsSuccess: [loadDocumentsSuccessCallback],
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

    // TODO: check that the expected side effect has happened.
    expect(onLoadDocumentsSuccessCallback1).toHaveBeenCalled();
    expect(onLoadDocumentsErrorCallback1).not.toHaveBeenCalled();
  });
});
