// TODO(LAS-399): Figure out proper project build structure to prevent lint error here
import { SeparatorTextChunker } from "../../../../src/transformation/document/text/separatorTextChunker";
import {
  getTestDocument,
  getTestDocumentFragment,
} from "../../../utils/testDocumentUtils";

const testFragment1Text = "This is a small fragment for simple testing";

const testFragment2Text = `
Contrary to popular belief, Lorem Ipsum is not simply random text. It has roots in a piece of 
classical Latin literature from 45 BC, making it over 2000 years old. Richard McClintock, a Latin 
professor at Hampden-Sydney College in Virginia, looked up one of the more obscure Latin words, 
consectetur, from a Lorem Ipsum passage, and going through the cites of the word in classical literature, 
discovered the undoubtable source.
`;

const testSmallDocument = getTestDocument({
  fragments: [getTestDocumentFragment({ content: testFragment1Text })],
});

const testLargerDocument = getTestDocument({
  fragments: [
    getTestDocumentFragment({ content: testFragment1Text }),
    getTestDocumentFragment({ content: testFragment2Text }),
  ],
});

describe("SeparatorTextChunker transformations", () => {
  test("default chunking (words) of single-fragment document; default chunk size and overlap", async () => {
    const chunker = new SeparatorTextChunker();
    const transformedDoc = await chunker.transformDocument(testSmallDocument);
    expect(transformedDoc.fragments.length).toEqual(1);
    expect(await transformedDoc.fragments[0].getContent()).toEqual(
      "This is a small fragment for simple testing"
    );
  });

  test("default chunking (words) of single-fragment document; small chunk size and no overlap", async () => {
    const chunker = new SeparatorTextChunker({
      chunkSizeLimit: 10,
      chunkOverlap: 0,
    });
    const transformedDoc = await chunker.transformDocument(testSmallDocument);
    expect(transformedDoc.fragments.length).toEqual(5);
    expect(await transformedDoc.fragments[0].getContent()).toEqual("This is a");
    expect(await transformedDoc.fragments[1].getContent()).toEqual("small");
    expect(await transformedDoc.fragments[2].getContent()).toEqual("fragment");
    expect(await transformedDoc.fragments[3].getContent()).toEqual(
      "for simple"
    );
    expect(await transformedDoc.fragments[4].getContent()).toEqual("testing");
  });

  test("default chunking (words) of single-fragment document; small chunk size and small overlap", async () => {
    const chunker = new SeparatorTextChunker({
      chunkSizeLimit: 10,
      chunkOverlap: 4,
    });
    const transformedDoc = await chunker.transformDocument(testSmallDocument);
    expect(transformedDoc.fragments.length).toEqual(5);
    expect(await transformedDoc.fragments[0].getContent()).toEqual("This is a");
    expect(await transformedDoc.fragments[1].getContent()).toEqual(
      "is a small"
    );
    expect(await transformedDoc.fragments[2].getContent()).toEqual("fragment");
    expect(await transformedDoc.fragments[3].getContent()).toEqual(
      "for simple"
    );
    expect(await transformedDoc.fragments[4].getContent()).toEqual("testing");
  });
});
