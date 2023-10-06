import { OpenAIEmbeddings } from "../../../src/transformation/embeddings/openAIEmbedding";
import {
  getTestDocument,
  getTestDocumentFragment,
} from "../../utils/testDocumentUtils";
import getEnvVar from "../../../src/utils/getEnvVar";
import { OpenAI } from "openai";

jest.mock("OpenAI");
jest.mock("../../../src/utils/getEnvVar", () => ({
  __esModule: true,
  default: jest.fn(),
}));
const mockedGetEnvVar = getEnvVar as jest.MockedFunction<typeof getEnvVar>;

describe("OpenAI Embeddings API key validation", () => {
  test("error thrown if no keys are available", () => {
    expect(() => new OpenAIEmbeddings()).toThrowError(
      "No OpenAI API key found for OpenAIEmbeddings"
    );
  });

  test("uses key from config when provided", () => {
    expect(
      () => new OpenAIEmbeddings({ apiKey: "test-config-key" })
    ).not.toThrow();
  });
});

describe("OpenAI Embeddings transformation, embedding single fragment", () => {
  beforeAll(() => {
    mockedGetEnvVar.mockReturnValue("test-env-key");
  });

  test("throws an error if the fragment exceeds max token size", async () => {
    const transformer = new OpenAIEmbeddings();
    const fragment = getTestDocumentFragment({ content: "word".repeat(8192) });
    await expect(transformer.embedFragment(fragment)).rejects.toThrowError(
      "Fragment text exceeds max input tokens (8191) for model text-embedding-ada-002"
    );
  });
});
