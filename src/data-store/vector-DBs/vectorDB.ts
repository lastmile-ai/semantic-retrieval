import { JSONObject } from "../../common/jsonTypes";
import type { Document } from "../../document/document"
import { DocumentMetadataDB } from "../../document/metadata/documentMetadataDB";

export type EmbeddingVector = number[] & {
  /**
   * Number of dimensions in the vector, and min/max values for each dimension.
   * This can be used to normalize the vector or checking for out-of-bounds values.
   */
  extras?: {
    dimensions: number;
    min: number;
    max: number;
  };
};

export interface VectorDBQuery {
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
  // Document filters
  documentFilter?: JSONObject;

  topK?: number;
}

export interface VectorDBEmbeddingQuery extends VectorDBQuery {
  // The embedding to query
  embeddingVector: EmbeddingVector;
}

export interface VectorDBTextQuery extends VectorDBQuery {
  // The text to query
  text: string,
}

export abstract class VectorDB {
  metadataDB?: DocumentMetadataDB;

  constructor(metadataDB?: DocumentMetadataDB) {
    this.metadataDB = metadataDB;
  }

  static async fromDocuments(_documents: Document[], _metadataDB?: DocumentMetadataDB): Promise<VectorDB> {
    throw new Error("VectorDB implementation missing override");
  }

  abstract addDocuments(documents: Document[]): Promise<void>;

  abstract query(query: VectorDBQuery): Promise<Document[]>;
}
