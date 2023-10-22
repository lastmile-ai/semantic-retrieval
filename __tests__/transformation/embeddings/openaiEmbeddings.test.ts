import { OpenAIEmbeddings } from "../../../src/transformation/embeddings/openAIEmbeddings";
import {
  getTestDocument,
  getTestDocumentFragment,
} from "../../__utils__/testDocumentUtils";
import getEnvVar from "../../../src/utils/getEnvVar";
import { OpenAI } from "openai";
import {
  CreateEmbeddingResponse,
  EmbeddingCreateParams,
} from "openai/resources/embeddings";
import { JSONObject } from "../../../src/common/jsonTypes";

function mockOpenAiEmbeddingsCreation(
  body: EmbeddingCreateParams
): CreateEmbeddingResponse {
  const embeddingRes = {
    model: "text-embedding-ada-002",
    object: "embedding",
    usage: {
      prompt_tokens: 1000,
      total_tokens: 2000,
    },
  };
  if (typeof body.input === "string") {
    return {
      ...embeddingRes,
      data: [
        {
          embedding: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
          index: 0,
          object: "embedding",
        },
      ],
    };
  } else if (Array.isArray(body.input)) {
    return {
      ...embeddingRes,
      data: (body.input as string[]).map((_input, idx) => {
        return {
          embedding: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
          index: idx,
          object: "embedding",
        };
      }),
    };
  } else {
    throw new Error("Unsupported input type for embeddings creation");
  }
}

const mockEmbeddingsCreate = jest
  .fn()
  .mockImplementation(mockOpenAiEmbeddingsCreation);

jest.mock("openai", () => {
  const originalModule = jest.requireActual("openai");

  return {
    ...originalModule,
    OpenAI: jest.fn().mockImplementation(() => ({
      embeddings: {
        create: mockEmbeddingsCreate,
      },
    })),
  };
});

jest.mock("../../../src/utils/getEnvVar");

const mockedOpenAI = OpenAI as jest.MockedClass<typeof OpenAI>;
const mockedGetEnvVar = getEnvVar as jest.MockedFunction<typeof getEnvVar>;

describe("OpenAI Embeddings API key validation", () => {
  test("error thrown if no keys are available", () => {
    expect(() => new OpenAIEmbeddings()).toThrowError(
      "No OpenAI API key found for OpenAIEmbeddings"
    );
  });

  test("uses key from config when provided", () => {
    mockedGetEnvVar.mockReturnValue("test-env-key");
    expect(
      () => new OpenAIEmbeddings({ apiKey: "test-config-key" })
    ).not.toThrow();
    expect(mockedOpenAI).toHaveBeenCalledWith({ apiKey: "test-config-key" });
  });

  test("uses key from environment when no config key provided", () => {
    mockedGetEnvVar.mockReturnValue("test-env-key");
    expect(() => new OpenAIEmbeddings()).not.toThrow();
    expect(mockedOpenAI).toHaveBeenCalledWith({ apiKey: "test-env-key" });
  });
});

describe("OpenAI Embeddings transformation, embedding text", () => {
  beforeAll(() => {
    mockedGetEnvVar.mockReturnValue("test-env-key");
  });

  beforeEach(() => {
    mockEmbeddingsCreate.mockClear();
  });

  test("throws an error if the text exceeds max token size", async () => {
    const transformer = new OpenAIEmbeddings();
    // "word" is 1 token in text-embedding-ada-002
    await expect(transformer.embed("word".repeat(8192))).rejects.toThrowError(
      `Text encoded length 8192 exceeds max input tokens (8191) for model text-embedding-ada-002`
    );
  });

  test("embeds a fragment with correct data", async () => {
    const transformer = new OpenAIEmbeddings();
    const embedding = await transformer.embed(
      "This is some example text to embed",
      { fragmentId: "test-fragment-id" }
    );
    expect(embedding.metadata.fragmentId).toBe("test-fragment-id");
    expect(embedding.metadata.model).toBe("text-embedding-ada-002");
    expect((embedding.metadata.usage as JSONObject).prompt_tokens).toBe(1000);
    expect((embedding.metadata.usage as JSONObject).total_tokens).toBe(2000);

    expect(mockEmbeddingsCreate).toHaveBeenCalledTimes(1);
    expect(mockEmbeddingsCreate).toHaveBeenCalledWith({
      input: "This is some example text to embed",
      model: "text-embedding-ada-002",
    });
  });
});

describe("OpenAI Embeddings transformation, embedding multiple fragments for multiple Documents", () => {
  beforeAll(() => {
    mockedGetEnvVar.mockReturnValue("test-env-key");
  });

  beforeEach(() => {
    mockEmbeddingsCreate.mockClear();
  });

  test("throws an error if any fragment exceeds max token size", async () => {
    const transformer = new OpenAIEmbeddings();
    const tooBigFragment = getTestDocumentFragment({
      content: "word".repeat(8192),
    });
    const testDocuments = [
      getTestDocument(),
      getTestDocument({
        fragments: [getTestDocumentFragment(), tooBigFragment],
      }),
    ];

    await expect(
      transformer.transformDocuments(testDocuments)
    ).rejects.toThrowError(
      `Fragment ${tooBigFragment.fragmentId} encoded length 8192 exceeds max input tokens (8191) for model text-embedding-ada-002`
    );
  });

  test("embeds all fragments with correct data in one batch if they fit", async () => {
    const transformer = new OpenAIEmbeddings();
    const testFragments = [
      getTestDocumentFragment({
        content: "This is the first fragment for the first document",
      }),
      getTestDocumentFragment({
        content: "This is the second fragment for the first document",
      }),
      getTestDocumentFragment({
        content: "This is the only fragment for the second document",
      }),
      getTestDocumentFragment({
        content: "This is the first fragment for the third document",
      }),
      getTestDocumentFragment({
        content: "This is the second fragment for the third document",
      }),
      getTestDocumentFragment({
        content: "This is the third fragment for the third document",
      }),
    ];
    const testDocuments = [
      getTestDocument({
        fragments: [testFragments[0], testFragments[1]],
      }),
      getTestDocument({
        fragments: [testFragments[2]],
      }),
      getTestDocument({
        fragments: [testFragments[3], testFragments[4], testFragments[5]],
      }),
    ];

    const embeddings = await transformer.transformDocuments(testDocuments);
    expect(embeddings.length).toBe(6);
    for (const [idx, embedding] of embeddings.entries()) {
      expect(embedding.metadata.fragmentId).toBe(testFragments[idx].fragmentId);
      expect(embedding.metadata.documentId).toBe(testFragments[idx].documentId);
      expect(embedding.metadata.model).toBe("text-embedding-ada-002");
      expect(embedding.metadata.usage).toBeUndefined();
    }

    expect(mockEmbeddingsCreate).toHaveBeenCalledTimes(1);
    expect(mockEmbeddingsCreate).toHaveBeenCalledWith({
      input: [
        "This is the first fragment for the first document",
        "This is the second fragment for the first document",
        "This is the only fragment for the second document",
        "This is the first fragment for the third document",
        "This is the second fragment for the third document",
        "This is the third fragment for the third document",
      ],
      model: "text-embedding-ada-002",
    });
  });

  test("embeds all fragments with correct data in multiple batches if they are large", async () => {
    const transformer = new OpenAIEmbeddings();

    const fragmentContents = [
      "word".repeat(4000),
      "word".repeat(4000),
      "word".repeat(191),
      "word".repeat(8000),
      "word".repeat(4000),
      "word".repeat(4192),
    ];
    const testFragments = [
      getTestDocumentFragment({
        content: fragmentContents[0],
      }),
      getTestDocumentFragment({
        content: fragmentContents[1],
      }),
      getTestDocumentFragment({
        content: fragmentContents[2],
      }),
      getTestDocumentFragment({
        content: fragmentContents[3],
      }),
      getTestDocumentFragment({
        content: fragmentContents[4],
      }),
      getTestDocumentFragment({
        content: fragmentContents[5],
      }),
    ];
    const testDocuments = [
      getTestDocument({
        fragments: [testFragments[0], testFragments[1]],
      }),
      getTestDocument({
        fragments: [testFragments[2]],
      }),
      getTestDocument({
        fragments: [testFragments[3], testFragments[4], testFragments[5]],
      }),
    ];

    const embeddings = await transformer.transformDocuments(testDocuments);
    expect(embeddings.length).toBe(6);
    for (const [idx, embedding] of embeddings.entries()) {
      expect(embedding.metadata.fragmentId).toBe(testFragments[idx].fragmentId);
      expect(embedding.metadata.documentId).toBe(testFragments[idx].documentId);
      expect(embedding.metadata.model).toBe("text-embedding-ada-002");
    }

    expect(mockEmbeddingsCreate).toHaveBeenCalledTimes(4);
    expect(mockEmbeddingsCreate).toHaveBeenCalledWith({
      input: [fragmentContents[0], fragmentContents[1], fragmentContents[2]],
      model: "text-embedding-ada-002",
    });
    expect(mockEmbeddingsCreate).toHaveBeenCalledWith({
      input: [fragmentContents[3]],
      model: "text-embedding-ada-002",
    });
    expect(mockEmbeddingsCreate).toHaveBeenCalledWith({
      input: [fragmentContents[4]],
      model: "text-embedding-ada-002",
    });
    expect(mockEmbeddingsCreate).toHaveBeenCalledWith({
      input: [fragmentContents[5]],
      model: "text-embedding-ada-002",
    });
  });
});
