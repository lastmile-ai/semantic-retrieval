import { SeparatorTextChunker } from "../../../../src/transformation/document/text/separatorTextChunker";
import { Document } from "../../../../src/document/document";
import {
  getTestDocument,
  getTestDocumentFragment,
} from "../../../__utils__/testDocumentUtils";

const testFragment1Text = "This is a small fragment for simple testing";

const testFragment2Text = `
This is a longer fragment for more complex testing. It has multiple lines and punctuation.
Contrary to popular belief, Lorem Ipsum is not simply random text. It has roots in a piece of \
classical Latin literature from 45 BC, making it over 2000 years old. Richard McClintock, a Latin \
professor at Hampden-Sydney College in Virginia, looked up one of the more obscure Latin words, \
consectetur, from a Lorem Ipsum passage, and going through the cites of the word in classical literature, \
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

async function validateFragmentChunks(
  document: Document,
  chunkSizeLimit: number,
  expectedChunks: string[]
) {
  expect(document.fragments.length).toEqual(expectedChunks.length);
  for (let i = 0; i < expectedChunks.length; i++) {
    const content = await document.fragments[i].getContent();
    expect(content.length).toBeLessThanOrEqual(chunkSizeLimit);
    expect(content).toEqual(expectedChunks[i]);
  }
}

describe("SeparatorTextChunker transformations", () => {
  test("default chunking (words) of single-fragment document; default chunk size and default overlap", async () => {
    const chunker = new SeparatorTextChunker();
    const transformedDoc = await chunker.transformDocument(testSmallDocument);
    await validateFragmentChunks(transformedDoc, 500, [
      "This is a small fragment for simple testing",
    ]);
  });

  test("default chunking (words) of single-fragment document; small chunk size and no overlap", async () => {
    const chunker = new SeparatorTextChunker({
      chunkSizeLimit: 10,
      chunkOverlap: 0,
    });
    const transformedDoc = await chunker.transformDocument(testSmallDocument);
    await validateFragmentChunks(transformedDoc, 10, [
      "This is a",
      "small",
      "fragment",
      "for simple",
      "testing",
    ]);
  });

  test("default chunking (words) of single-fragment document; small chunk size and small overlap", async () => {
    const chunker = new SeparatorTextChunker({
      chunkSizeLimit: 10,
      chunkOverlap: 2,
    });
    const transformedDoc = await chunker.transformDocument(testSmallDocument);
    await validateFragmentChunks(transformedDoc, 10, [
      "This is a",
      "a small",
      "fragment",
      "for simple",
      "testing",
    ]);
  });

  test("default chunking (words) of single-fragment document; small chunk size and large overlap", async () => {
    const chunker = new SeparatorTextChunker({
      chunkSizeLimit: 12,
      chunkOverlap: 8,
    });
    const transformedDoc = await chunker.transformDocument(testSmallDocument);
    await validateFragmentChunks(transformedDoc, 12, [
      "This is a",
      "is a small",
      "fragment for",
      "for simple",
      "testing",
    ]);
  });

  test("default chunking (words) of single-fragment document; large chunk size and no overlap", async () => {
    const chunker = new SeparatorTextChunker({
      chunkSizeLimit: 30,
      chunkOverlap: 0,
    });
    const transformedDoc = await chunker.transformDocument(testSmallDocument);
    await validateFragmentChunks(transformedDoc, 30, [
      "This is a small fragment for",
      "simple testing",
    ]);
  });

  test("default chunking (words) of single-fragment document; large chunk size and overlap", async () => {
    const chunker = new SeparatorTextChunker({
      chunkSizeLimit: 30,
      chunkOverlap: 15,
    });
    const transformedDoc = await chunker.transformDocument(testSmallDocument);
    await validateFragmentChunks(transformedDoc, 30, [
      "This is a small fragment for",
      "fragment for simple testing",
    ]);
  });

  test("character chunking of single-fragment document; default chunk size and default overlap", async () => {
    const chunker = new SeparatorTextChunker({
      separator: "",
    });
    const transformedDoc = await chunker.transformDocument(testSmallDocument);
    await validateFragmentChunks(transformedDoc, 500, [
      "This is a small fragment for simple testing",
    ]);
  });

  test("character chunking of single-fragment document; small chunk size and overlap", async () => {
    const chunker = new SeparatorTextChunker({
      separator: "",
      chunkSizeLimit: 10,
      chunkOverlap: 2,
    });
    const transformedDoc = await chunker.transformDocument(testSmallDocument);
    await validateFragmentChunks(transformedDoc, 10, [
      "This is a",
      "a small fr",
      "fragment f",
      "for simpl",
      "ple testin",
      "ing",
    ]);
  });

  test("character chunking (words) of single-fragment document; large chunk size and overlap", async () => {
    const chunker = new SeparatorTextChunker({
      separator: "",
      chunkSizeLimit: 30,
      chunkOverlap: 15,
    });
    const transformedDoc = await chunker.transformDocument(testSmallDocument);
    await validateFragmentChunks(transformedDoc, 30, [
      "This is a small fragment for s",
      "fragment for simple testing",
    ]);
  });

  test("default chunking (words) of multi-fragment document; default chunk size and default overlap", async () => {
    const chunker = new SeparatorTextChunker();
    const transformedDoc = await chunker.transformDocument(testLargerDocument);
    await validateFragmentChunks(transformedDoc, 500, [
      "This is a small fragment for simple testing",
      `This is a longer fragment for more complex testing. It has multiple lines and punctuation. \
Contrary to popular belief, Lorem Ipsum is not simply random text. It has roots in a piece of classical Latin literature from 45 BC, \
making it over 2000 years old. Richard McClintock, a Latin professor at Hampden-Sydney College in Virginia, looked up one of the more \
obscure Latin words, consectetur, from a Lorem Ipsum passage, and going through the cites of the word in classical literature, discovered the`,
      "Lorem Ipsum passage, and going through the cites of the word in classical literature, discovered the undoubtable source.",
    ]);
  });

  test("default chunking (words) of multi-fragment document; small chunk size and small overlap", async () => {
    const chunker = new SeparatorTextChunker({
      chunkSizeLimit: 70,
      chunkOverlap: 20,
    });
    const transformedDoc = await chunker.transformDocument(testLargerDocument);
    await validateFragmentChunks(transformedDoc, 70, [
      "This is a small fragment for simple testing",
      "This is a longer fragment for more complex testing. It has multiple",
      "It has multiple lines and punctuation. Contrary to popular belief,",
      "to popular belief, Lorem Ipsum is not simply random text. It has roots",
      "text. It has roots in a piece of classical Latin literature from 45",
      "literature from 45 BC, making it over 2000 years old. Richard",
      "years old. Richard McClintock, a Latin professor at Hampden-Sydney",
      "at Hampden-Sydney College in Virginia, looked up one of the more",
      "up one of the more obscure Latin words, consectetur, from a Lorem",
      "from a Lorem Ipsum passage, and going through the cites of the word in",
      "cites of the word in classical literature, discovered the undoubtable",
      "the undoubtable source.",
    ]);
  });

  test("character chunking of multi-fragment document; default chunk size and default overlap", async () => {
    const chunker = new SeparatorTextChunker({
      separator: "",
    });
    const transformedDoc = await chunker.transformDocument(testLargerDocument);
    await validateFragmentChunks(transformedDoc, 500, [
      "This is a small fragment for simple testing",
      `This is a longer fragment for more complex testing. It has multiple lines and punctuation. \
Contrary to popular belief, Lorem Ipsum is not simply random text. It has roots in a piece of classical Latin literature from 45 BC, \
making it over 2000 years old. Richard McClintock, a Latin professor at Hampden-Sydney College in Virginia, looked up one of the more \
obscure Latin words, consectetur, from a Lorem Ipsum passage, and going through the cites of the word in classical literature, discovered the`,
      "Lorem Ipsum passage, and going through the cites of the word in classical literature, discovered the undoubtable source.",
    ]);
  });
});
