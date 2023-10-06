import {
  DocumentFragmentEmbeddingsTransformer,
  DocumentFragmentVectorEmbedding,
} from "./embeddings";
import { Document, DocumentFragment } from "../../document/document";
import { type ClientOptions as OpenAIClientOptions, OpenAI } from "openai";
import { JSONObject } from "../../common/jsonTypes";
import { TiktokenModel, encoding_for_model } from "@dqbd/tiktoken";
import { requestWithThrottleBackoff } from "../../utils/axiosUtils";

interface OpenAIEmbeddingsConfig extends OpenAIClientOptions {
  apiKey?: string;
}

type EmbedFragmentData = {
  fragmentId: string;
  text: string;
};

/**
 * Transforms Documents into VectorEmbeddings using OpenAI's embedding model API. Each
 * fragment of a Document is embedded as a separate VectorEmbedding and should not exceed
 * the embedding model's max input tokens (8191 for text-embedding-ada-002).
 */
export class OpenAIEmbeddings extends DocumentFragmentEmbeddingsTransformer {
  // For now, only support the text-embedding-ada-002 model to allow optimized batching.
  // There is no clear information on / tokenization support for other embedding models
  model = "text-embedding-ada-002" as TiktokenModel;

  private client: OpenAI;

  constructor(config?: OpenAIEmbeddingsConfig) {
    super();

    const apiKey = config?.apiKey ?? process.env.OPENAI_API_KEY;
    if (!apiKey) {
      throw new Error("No OpenAI API key found for OpenAIEmbeddings");
    }

    this.client = new OpenAI({ ...config, apiKey });
  }

  async embedFragment(
    fragment: DocumentFragment
  ): Promise<DocumentFragmentVectorEmbedding> {
    const fragmentData = {
      fragmentId: fragment.fragmentId,
      text: await fragment.getContent(),
    };
    return (await this.createEmbeddings([fragmentData]))[0];
  }

  async transformDocuments(
    documents: Document[]
  ): Promise<DocumentFragmentVectorEmbedding[]> {
    const embeddings: DocumentFragmentVectorEmbedding[] = [];

    // We'll send the actual requests in batches of 5 in case there are lots of large documents
    let requestBatch: EmbedFragmentData[][] = [];
    const REQUEST_BATCH_SIZE = 5;

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

        // TODO: Handle this for other models when they are supported
        if (currentFragmentEncoding.length > 8191) {
          throw new Error(
            `Fragment text exceeds max input tokens for model ${this.model} (8191 for text-embedding-ada-002)`
          );
        }

        if (currentTextBatchSize + currentFragmentEncoding.length > 8191) {
          requestBatch.push(currentTextBatch);

          if (requestBatch.length === REQUEST_BATCH_SIZE) {
            const embeddingPromises = requestBatch.map((batch) =>
              this.createEmbeddings(batch)
            );
            embeddings.push(...(await Promise.all(embeddingPromises)).flat());
            currentTextBatch = [];
            currentTextBatchSize = 0;
            requestBatch = [];
          }
        }

        currentTextBatch.push(currentFragmentData);
        currentTextBatchSize += currentFragmentEncoding.length;
        currentDocumentFragmentIdx++;
      }

      documentIdx++;
    }

    // Handle remaining requests after all documents have been processed
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
  ): Promise<DocumentFragmentVectorEmbedding[]> {
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
    return data.map((embedding, idx) => ({
      fragmentId: fragments[idx].fragmentId,
      vector: embedding.embedding,
      metadata: {
        ...metadata,
        model: this.model,
        usage,
      },
      attributes: {},
    }));
  }
}
