import {
  Index,
  Pinecone,
  PineconeRecord,
  RecordMetadata,
} from "@pinecone-database/pinecone";
import type { Document } from "../../document/document";

import {
  VectorDB,
  VectorDBConfig,
  VectorDBQuery,
  isEmbeddingQuery,
  VectorDBTextQuery,
} from "./vectorDB";
import getEnvVar from "../../utils/getEnvVar";
import { requestWithThrottleBackoff } from "../../utils/axiosUtils";
import { flatten, unflatten } from "safe-flat";
import { v4 as uuid } from "uuid";
import { VectorEmbedding } from "../../transformation/embeddings/embeddings";
import { JSONObject } from "../../common/jsonTypes";
import {
  AddDocumentsToVectorDBEvent,
  QueryVectorDBEvent,
} from "../../utils/callbacks";

export type PineconeVectorDBConfig = VectorDBConfig & {
  indexName: string;
  apiKey?: string;
  environment?: string;
  namespace?: string;
  vectorsPerRequest?: number;
  maxParallelBatchRequests?: number;
};

/**
 * A VectorDB backed by an existing Pinecone index. The index referenced by this VectorDB should
 * be created beforehand using the Pinecone console, UI, or API.
 */
export class PineconeVectorDB extends VectorDB {
  client: Pinecone;
  index: Index;
  namespace: Index<RecordMetadata>;

  // Pinecone recommends a limit of 100 vectors max per upsert request
  vectorsPerRequest = 100;
  maxParallelBatchRequests = 20;

  constructor(config: PineconeVectorDBConfig) {
    super(config.embeddings, config.metadataDB, config.callbackManager);

    const apiKey = config?.apiKey ?? getEnvVar("PINECONE_API_KEY");
    if (!apiKey) {
      throw new Error("No Pinecone API key found for PineconeVectorDB");
    }

    const environment =
      config?.environment ?? getEnvVar("PINECONE_ENVIRONMENT");
    if (!environment) {
      throw new Error("No Pinecone environment found for PineconeVectorDB");
    }

    this.vectorsPerRequest =
      config?.vectorsPerRequest ?? this.vectorsPerRequest;
    this.maxParallelBatchRequests =
      config?.maxParallelBatchRequests ?? this.maxParallelBatchRequests;

    this.client = new Pinecone({ apiKey, environment });
    this.index = this.client.index(config.indexName);

    // Default namespace is empty string
    this.namespace = this.index.namespace(config.namespace ?? "");

    // If the dimensions don't match, future operations will error out, so warn asap
    this.index.describeIndexStats().then((stats) => {
      if (
        stats.dimension != null &&
        stats.dimension !== this.embeddings.dimensions
      ) {
        console.error(
          `PineconeVectorDB embedding dimensions (${this.embeddings.dimensions}) do not match index dimensions (${stats.dimension})`
        );
      }
    });
  }

  static async fromDocuments(
    documents: Document[],
    config: PineconeVectorDBConfig
  ): Promise<VectorDB> {
    const instance = new this(config);
    await instance.addDocuments(documents);
    return instance;
  }

  // Sanitize metadata to ensure it is compatible with Pinecone's metadata.
  // Nested objects are flattened to a single level, and accessible via dot notation,
  // e.g. {key1: {keyA: 'test'}} becomes {key1.keyA: 'test'}.
  // Metadata is stored in Pinecone as key-value pairs, with supported value types:
  // String, Number, Boolean, List of String
  // https://docs.pinecone.io/docs/metadata-filtering
  // NOTE: Pinecone will index all metadata fields by default. Create the index
  // with a metadata_config to configure selective metadata indexing.
  // https://docs.pinecone.io/docs/manage-indexes#selective-metadata-indexing
  private sanitizeMetadata(
    unsanitizedMetadata: Record<string, unknown>
  ): RecordMetadata {
    const stringArrayMetadata: { [key: string]: string[] } = {};
    const mutableMetadata = { ...unsanitizedMetadata };

    // Recursively add string arrays to metadata first to prevent flattening in the next step
    // Each nested array's key is flattened to dot notation,
    // e.g. {key1: {keyA: ['test']}} becomes {key1.keyA: ['test']}
    const setStringArrayMetadataRecursive = (
      metadata: Record<string, unknown>,
      keyPath: string[] = []
    ) => {
      for (const [key, value] of Object.entries(metadata)) {
        const updatedKeyPath = [...keyPath, key];
        if (Array.isArray(value) && value.every((v) => typeof v === "string")) {
          stringArrayMetadata[updatedKeyPath.join(".")] = value;
          delete metadata[key];
        } else if (typeof value === "object") {
          setStringArrayMetadataRecursive(
            value as Record<string, unknown>,
            updatedKeyPath
          );
        }
      }
    };

    setStringArrayMetadataRecursive(mutableMetadata);

    const metadata: {
      [key: string]: string | number | boolean | string[];
    } = {
      ...(flatten(mutableMetadata) as Record<
        string,
        string | number | boolean
      >),
      ...stringArrayMetadata,
    };

    // Remove nulls since Pinecone does not support null values
    for (const [key, value] of Object.entries(metadata)) {
      if (value == null) {
        delete metadata[key];
      }
    }

    return metadata;
  }

  async addDocuments(documents: Document[]): Promise<void> {
    const embeddings = await this.embeddings.transformDocuments(documents);
    const pineconeVectors: PineconeRecord<RecordMetadata>[] = embeddings.map(
      (embedding) => ({
        id: uuid(),
        values: embedding.vector,
        metadata: this.sanitizeMetadata({
          ...embedding.metadata,
          ...embedding.attributes,
          text: embedding.text,
        }),
      })
    );

    let vectorIdx = 0;
    while (vectorIdx < pineconeVectors.length) {
      const requests: Promise<void>[] = [];
      for (let i = 0; i < this.maxParallelBatchRequests; i++) {
        const vectors = pineconeVectors.slice(
          vectorIdx,
          vectorIdx + this.vectorsPerRequest
        );
        if (vectors.length > 0) {
          requests.push(
            requestWithThrottleBackoff(() => this.namespace.upsert(vectors))
          );
          vectorIdx += this.vectorsPerRequest;
        }
      }
      await Promise.all(requests);
    }

    const event: AddDocumentsToVectorDBEvent = {
      name: "onAddDocumentsToVectorDB",
      documents,
    };
    await this.callbackManager?.runCallbacks(event);
  }

  async query(query: VectorDBQuery): Promise<VectorEmbedding[]> {
    let queryVector;
    if (isEmbeddingQuery(query)) {
      queryVector = query.embeddingVector.vector;
    } else {
      const text = (query as VectorDBTextQuery).text;
      queryVector = (await this.embeddings.embed(text)).vector;
    }

    const results = await this.namespace.query({
      includeMetadata: true,
      topK: query.topK,
      vector: queryVector,
      filter: query.metadataFilter,
    });

    const vectorEmbeddings = (results.matches ?? []).map((match) => {
      const metadata = unflatten({ ...match.metadata }) as JSONObject;
      const attributes = (metadata.attributes ?? {}) as JSONObject;
      const text = (metadata.text as string | undefined) ?? "";
      delete metadata["attributes"];
      delete metadata["text"];

      if (match.score) {
        metadata["retrievalScore"] = match.score;
      }

      return {
        vector: match.values,
        text,
        metadata,
        attributes,
      };
    });

    const event: QueryVectorDBEvent = {
      name: "onQueryVectorDB",
      query: query,
      vectorEmbeddings: vectorEmbeddings,
    };
    await this.callbackManager?.runCallbacks(event);

    return vectorEmbeddings;
  }
}
