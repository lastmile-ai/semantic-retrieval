// TODO(LAS-399): Figure out proper project build structure to prevent lint error here
import { RecursiveSeparatorTextChunker } from "../../../../src/transformation/document/text/recursiveSeparatorTextChunker";
import {
  getTestDocument,
  getTestDocumentFragment,
} from "../../../utils/testDocumentUtils";
import { validateFragmentChunks } from "../../testTransformationUtils";

const testFragment1Text = "This is a small fragment for simple testing";

const testFragment2Text = `
This is a longer fragment for more complex testing. It has paragraphs, lines and punctuation.
The next paragraph will discuss Lorem Ipsum text.

Lorem Ipsum is simply dummy text of the printing and typesetting industry. \
Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown \
printer took a galley of type and scrambled it to make a type specimen book. It has survived \
not only five centuries, but also the leap into electronic typesetting, remaining essentially \
unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum \
passages, and more recently with desktop publishing software like Aldus PageMaker including versions \
of Lorem Ipsum.

Contrary to popular belief, Lorem Ipsum is not simply random text. It has roots in a piece of \
classical Latin literature from 45 BC, making it over 2000 years old. Richard McClintock, a Latin \
professor at Hampden-Sydney College in Virginia, looked up one of the more obscure Latin words, \
consectetur, from a Lorem Ipsum passage, and going through the cites of the word in classical literature, \
discovered the undoubtable source.

That sure is interesting!
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

describe("RecursiveSeparatorTextChunker transformations", () => {
  test("default chunking of single-fragment document; default chunk size and default overlap", async () => {
    const chunker = new RecursiveSeparatorTextChunker();
    const transformedDoc = await chunker.transformDocument(testSmallDocument);
    await validateFragmentChunks(transformedDoc, 500, [
      "This is a small fragment for simple testing",
    ]);
  });

  test("default chunking of single-fragment document; small chunk size and no overlap", async () => {
    const chunker = new RecursiveSeparatorTextChunker({
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

  test("default chunking of single-fragment document; small chunk size and small overlap", async () => {
    const chunker = new RecursiveSeparatorTextChunker({
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

  test("default chunking of single-fragment document; small chunk size and large overlap", async () => {
    const chunker = new RecursiveSeparatorTextChunker({
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

  test("default chunking of single-fragment document; large chunk size and no overlap", async () => {
    const chunker = new RecursiveSeparatorTextChunker({
      chunkSizeLimit: 30,
      chunkOverlap: 0,
    });
    const transformedDoc = await chunker.transformDocument(testSmallDocument);
    await validateFragmentChunks(transformedDoc, 30, [
      "This is a small fragment for",
      "simple testing",
    ]);
  });

  test("default chunking of single-fragment document; large chunk size and overlap", async () => {
    const chunker = new RecursiveSeparatorTextChunker({
      chunkSizeLimit: 30,
      chunkOverlap: 15,
    });
    const transformedDoc = await chunker.transformDocument(testSmallDocument);
    await validateFragmentChunks(transformedDoc, 30, [
      "This is a small fragment for",
      "fragment for simple testing",
    ]);
  });

  test("sentence chunking of single-fragment document; default chunk size and default overlap", async () => {
    const chunker = new RecursiveSeparatorTextChunker({
      separators: [
        "\n\n",
        "\n",
        ". ",
        "? ",
        "! ",
        ".\n",
        "?\n",
        "!\n",
        " ",
        "",
      ],
    });
    const transformedDoc = await chunker.transformDocument(testSmallDocument);
    await validateFragmentChunks(transformedDoc, 500, [
      "This is a small fragment for simple testing",
    ]);
  });

  test("word chunking of single-fragment document; small chunk size and overlap", async () => {
    const chunker = new RecursiveSeparatorTextChunker({
      separators: [" "],
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

  test("character chunking of single-fragment document; large chunk size and overlap", async () => {
    const chunker = new RecursiveSeparatorTextChunker({
      separators: [""],
      chunkSizeLimit: 30,
      chunkOverlap: 15,
    });
    const transformedDoc = await chunker.transformDocument(testSmallDocument);
    await validateFragmentChunks(transformedDoc, 30, [
      "This is a small fragment for s",
      "fragment for simple testing",
    ]);
  });

  test("default chunking of multi-fragment document; default chunk size and default overlap", async () => {
    const chunker = new RecursiveSeparatorTextChunker();
    const transformedDoc = await chunker.transformDocument(testLargerDocument);
    await validateFragmentChunks(transformedDoc, 500, [
      "This is a small fragment for simple testing",
      `This is a longer fragment for more complex testing. It has paragraphs, lines and punctuation.\ 
The next paragraph will discuss Lorem Ipsum text.
Contrary to popular belief, Lorem Ipsum is not simply random text. It has roots in a piece of classical Latin literature from 45 BC, \
making it over 2000 years old. Richard McClintock, a Latin professor at Hampden-Sydney College in Virginia, looked up one of the more \
obscure Latin words, consectetur, from a Lorem Ipsum passage, and going through the cites of the word in classical literature, discovered the`,
      "Lorem Ipsum passage, and going through the cites of the word in classical literature, discovered the undoubtable source.",
    ]);
  });

  test("default chunking of multi-fragment document; small chunk size and small overlap", async () => {
    const chunker = new RecursiveSeparatorTextChunker({
      chunkSizeLimit: 80,
      chunkOverlap: 30,
    });
    const transformedDoc = await chunker.transformDocument(testLargerDocument);
    await validateFragmentChunks(transformedDoc, 70, [
      "This is a small fragment for simple testing",
      "This is a longer fragment for more complex testing. It has multiple",
      "It has paragraphs, lines and punctuation.\nContrary to popular belief,",
      "to popular belief, Lorem Ipsum is not simply random text. It has roots",
      "text. It has roots in a piece of classical Latin literature from 45",
      "literature from 45 BC, making it over 2000 years old. Richard",
      "years old. Richard McClintock, a Latin professor at Hampden-Sydney",
      "at Hampden-Sydney College in Virginia, looked up one of the more",
      "up one of the more obscure Latin words, consectetur, from a Lorem",
      "from a Lorem Ipsum passage, and going through the cites of the word in",
      "cites of the word in classical literature, discovered the undoubtable",
      "the undoubtable source.",
      "test",
      "test",
      "test",
      "test",
      "test",
      "test",
      "test",
      "test",
      "test",
      "test",
      "test",
      "test",
    ]);
  });

  test("sentence chunking of multi-fragment document; default chunk size and default overlap", async () => {
    const chunker = new RecursiveSeparatorTextChunker({
      separators: [
        "\n\n",
        "\n",
        ". ",
        "? ",
        "! ",
        ".\n",
        "?\n",
        "!\n",
        " ",
        "",
      ],
    });
    const transformedDoc = await chunker.transformDocument(testLargerDocument);
    await validateFragmentChunks(transformedDoc, 500, [
      "This is a small fragment for simple testing",
      `This is a longer fragment for more complex testing. It has paragraphs, lines and punctuation.
The next paragraph will discuss Lorem Ipsum text.`,
      `Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text \
ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived \
not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged`,
      `It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum \
passages, and more recently with desktop publishing software like Aldus PageMaker including versions \
of Lorem Ipsum.`,
      `Contrary to popular belief, Lorem Ipsum is not simply random text. It has roots in a piece of classical Latin literature from 45 BC, \
making it over 2000 years old. Richard McClintock, a Latin professor at Hampden-Sydney College in Virginia, looked up one of the more \
obscure Latin words, consectetur, from a Lorem Ipsum passage, and going through the cites of the word in classical literature, discovered the \
undoubtable source.

That sure is interesting!`,
    ]);
  });
});
