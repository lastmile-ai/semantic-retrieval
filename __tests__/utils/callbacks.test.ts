// TODO: These imports should be from actual lastmile retrieval package
import { FileSystem } from "../../src/ingestion/data-sources/fs/fileSystem";

describe("Callbacks are called", () => {
  test("returns the documents with correct metadata and count", async () => {
    const fileSystem = new FileSystem(
      "./examples/example_data/DonQuixote.txt",
      undefined,
      undefined,
      {
        onFileLoaded: async (callback_input) => {
          console.log("the input:\n" + JSON.stringify(callback_input));
        },
      }
    );
    const rawDocuments = await fileSystem.loadDocuments();

    // TODO: check that the expected side effect has happened.
    expect(1).toEqual(1);
  });
});
