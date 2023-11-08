import { DocumentEmbeddingsTransformer, VectorEmbedding } from "./embeddings";
import { Document } from "../../document/document";
import { type ClientOptions as OpenAIClientOptions, OpenAI } from "openai";
import { JSONObject } from "../../common/jsonTypes";
import { TiktokenModel, encoding_for_model } from "@dqbd/tiktoken";
import { requestWithThrottleBackoff } from "../../utils/axiosUtils";
import getEnvVar from "../../utils/getEnvVar";

interface OpenAIEmbeddingsConfig extends OpenAIClientOptions {
  apiKey?: string;

  // Max parallel requests when performing batch embedding requests
  maxParallelBatchRequests?: number;
}

type EmbedFragmentData = {
  documentId: string;
  fragmentId: string;
  text: string;
};

// For now, only support the text-embedding-ada-002 model to allow optimized batching.
// There is no clear information on / tokenization support for other embedding models
const DEFAULT_MODEL = "text-embedding-ada-002";

const MODEL_DIMENSIONS = {
  "text-embedding-ada-002": 1536,
};

/**
 * Transforms Documents into VectorEmbeddings using OpenAI's embedding model API. Each
 * fragment of a Document is embedded as a separate VectorEmbedding and should not exceed
 * the embedding model's max input tokens (8191 for text-embedding-ada-002).
 */
export class OpenAIEmbeddings extends DocumentEmbeddingsTransformer {
  model = DEFAULT_MODEL as TiktokenModel;

  maxParallelBatchRequests = 20;

  // TODO: Handle this for other models when they are supported
  maxEncodingLength = 8191;

  private client: OpenAI;

  constructor(config?: OpenAIEmbeddingsConfig) {
    super(MODEL_DIMENSIONS[DEFAULT_MODEL]);

    const apiKey = config?.apiKey ?? getEnvVar("OPENAI_API_KEY");
    if (!apiKey) {
      throw new Error("No OpenAI API key found for OpenAIEmbeddings");
    }

    this.maxParallelBatchRequests =
      config?.maxParallelBatchRequests ?? this.maxParallelBatchRequests;

    this.client = new OpenAI({ ...config, apiKey });
  }

  async embed(text: string, metadata?: JSONObject): Promise<VectorEmbedding> {
    const encoding = encoding_for_model(this.model);
    const textEncoding = encoding.encode(text);

    if (textEncoding.length > this.maxEncodingLength) {
      encoding.free();
      throw new Error(
        `Text encoded length ${textEncoding.length} exceeds max input tokens (${this.maxEncodingLength}) for model ${this.model}`
      );
    }

    encoding.free();

    const embeddingRes = await this.client.embeddings.create({
      input: text,
      model: this.model,
    });

    const { data, usage, ...embeddingMetadata } = embeddingRes;

    return {
      vector: data[0].embedding,
      text,
      metadata: {
        ...embeddingMetadata,
        usage: usage as unknown as JSONObject,
        ...metadata,
        model: this.model,
      },
      attributes: {},
    };
  }

  async transformDocuments(documents: Document[]): Promise<VectorEmbedding[]> {
    const embeddings: VectorEmbedding[] = [];

    // We'll send the actual requests in batches of maxParallelBatchRequests size
    // in case there are lots of large documents
    let requestBatch: EmbedFragmentData[][] = [];

    let currentTextBatch: EmbedFragmentData[] = [];
    let currentTextBatchSize = 0;
    let documentIdx = 0;

    const encoding = encoding_for_model(this.model);

    // openai supports batch embedding creation by sending a list of texts to embed in a
    // single request, as long as the total length does not exceed the max input tokens
    // for the model (8191 for text-embedding-ada-002). So, here we batch all fragment
    // text across all documents to optimize individual requests in batches of fragment texts.
    while (documentIdx < documents.length) {
      const currentDocument = documents[documentIdx];

      // TODO: Revisit if this is problematic for large documents; we can always move the
      // fragment getContent logic into a loop below (at the cost of more code complexity)
      const currentDocumentFragments = await Promise.all(
        currentDocument.fragments.map(async (fragment) => ({
          documentId: fragment.documentId,
          fragmentId: fragment.fragmentId,
          text: await fragment.getContent(),
        }))
      );

      let currentDocumentFragmentIdx = 0;

      while (currentDocumentFragmentIdx < currentDocumentFragments.length) {
        const currentFragmentData =
          currentDocumentFragments[currentDocumentFragmentIdx];
        const currentFragmentEncoding = encoding.encode(
          currentFragmentData.text
        );

        if (currentFragmentEncoding.length > this.maxEncodingLength) {
          encoding.free();
          throw new Error(
            `Fragment ${currentFragmentData.fragmentId} encoded length ${currentFragmentEncoding.length} exceeds max input tokens (${this.maxEncodingLength}) for model ${this.model}`
          );
        }

        if (
          currentTextBatchSize + currentFragmentEncoding.length >
          this.maxEncodingLength
        ) {
          requestBatch.push(currentTextBatch);

          if (requestBatch.length === this.maxParallelBatchRequests) {
            const embeddingPromises = requestBatch.map((batch) =>
              this.createEmbeddings(batch)
            );
            embeddings.push(...(await Promise.all(embeddingPromises)).flat());
            requestBatch = [];
          }

          currentTextBatch = [];
          currentTextBatchSize = 0;
        }

        currentTextBatch.push(currentFragmentData);
        currentTextBatchSize += currentFragmentEncoding.length;
        currentDocumentFragmentIdx++;
      }

      documentIdx++;
    }

    // Handle remaining requests after all documents have been processed
    if (currentTextBatch.length > 0) {
      requestBatch.push(currentTextBatch);
    }

    if (requestBatch.length > 0) {
      const embeddingPromises = requestBatch.map((batch) =>
        this.createEmbeddings(batch)
      );
      embeddings.push(...(await Promise.all(embeddingPromises)).flat());
    }

    encoding.free();
    return embeddings;
  }

  private async createEmbeddings(
    fragments: EmbedFragmentData[]
  ): Promise<VectorEmbedding[]> {
    const input = fragments.map((fragment) => fragment.text);
    const embeddingRes = await requestWithThrottleBackoff(() =>
      this.client.embeddings.create({
        input,
        model: this.model,
      })
    );
    const { data, usage: metadataUsage, ...metadata } = embeddingRes;
    let usage: JSONObject | undefined;
    // Including total usage for batch request in each embedding could be confusing, so
    // just add it if a single fragment was embedded.
    // TODO: Track the usage when we add eval/usage tracking to the SDK
    if (fragments.length === 1) {
      // usage can't be typed as JSONObject since its keys are static in the interface
      usage = metadataUsage as unknown as JSONObject;
    }

    return data.map((embedding, idx) => {
      const embeddingMetadata: JSONObject = {
        ...metadata,
        fragmentId: fragments[idx].fragmentId,
        documentId: fragments[idx].documentId,
        model: this.model,
      };

      if (usage) {
        embeddingMetadata["usage"] = usage;
      }

      return {
        vector: embedding.embedding,
        text: fragments[idx].text,
        metadata: embeddingMetadata,
      };
    });
  }
}
