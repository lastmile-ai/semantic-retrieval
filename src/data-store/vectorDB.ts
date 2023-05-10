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
  mode:
    | "default"
    | "sparse"
    | "hybrid"
    | "dense"
    | "svm"
    | "logistic"
    | "linear";

  // The embedding to query
  embeddingVector: EmbeddingVector;

  // Metadata filtering, such as https://docs.pinecone.io/docs/metadata-filtering
  metadataFilter: { [key: string]: any };
  // Document filters
  documentFilter: { [key: string]: any };

  topK: number;
}

export interface VectorDB {}
