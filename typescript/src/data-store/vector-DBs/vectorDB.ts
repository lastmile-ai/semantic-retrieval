import { JSONObject } from "../../common/jsonTypes";
import type { Document } from "../../document/document";
import { DocumentMetadataDB } from "../../document/metadata/documentMetadataDB";
import {
  EmbeddingsTransformer,
  VectorEmbedding,
} from "../../transformation/embeddings/embeddings";
import { CallbackManager, Traceable } from "../../utils/callbacks";

export type VectorDBQuery = VectorDBEmbeddingQuery | VectorDBTextQuery;

export interface VectorDBBaseQuery {
  // TODO: saqadri - revisit
  mode?:
    | "default"
    | "sparse"
    | "hybrid"
    | "dense"
    | "svm"
    | "logistic"
    | "linear";

  // Metadata filtering, such as https://docs.pinecone.io/docs/metadata-filtering
  metadataFilter?: JSONObject;

  // The top K most similar results to return
  topK: number;
}

export interface VectorDBEmbeddingQuery extends VectorDBBaseQuery {
  // The embedding to query
  embeddingVector: VectorEmbedding;
}

export function isEmbeddingQuery(
  query: VectorDBQuery
): query is VectorDBEmbeddingQuery {
  return (query as VectorDBEmbeddingQuery).embeddingVector != null;
}

export interface VectorDBTextQuery extends VectorDBBaseQuery {
  // The text to query
  text: string;
}

export function isTextQuery(query: VectorDBQuery): query is VectorDBTextQuery {
  return (query as VectorDBTextQuery).text != null;
}

export interface VectorDBConfig {
  embeddings: EmbeddingsTransformer;
  metadataDB: DocumentMetadataDB;
  callbackManager?: CallbackManager;
}

/**
 * A VectorDB is a database that stores and retrieves Documents by their vector embeddings.
 * VectorDBs can be used to store and retrieve Documents via vector similarity queries.
 * The VectorDB will use the provided EmbeddingsTransformer to transform Documents into
 * vector embeddings to store, and to transform queries into vector embeddings to query.
 * Please make sure that the underlying VectorDB implementation supports the dimensionality
 * of the embeddings produced by the provided EmbeddingsTransformer.
 */
export abstract class VectorDB implements VectorDBConfig, Traceable {
  embeddings: EmbeddingsTransformer;
  metadataDB: DocumentMetadataDB;
  callbackManager?: CallbackManager;

  constructor(
    embeddings: EmbeddingsTransformer,
    metadataDB: DocumentMetadataDB,
    callbackManager?: CallbackManager
  ) {
    this.embeddings = embeddings;
    this.metadataDB = metadataDB;
    this.callbackManager = callbackManager;
  }

  static async fromDocuments(
    _documents: Document[],
    _config: VectorDBConfig
  ): Promise<VectorDB> {
    throw new Error("VectorDB implementation missing override");
  }

  abstract addDocuments(documents: Document[]): Promise<void>;

  abstract query(query: VectorDBQuery): Promise<VectorEmbedding[]>;
}
