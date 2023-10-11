import { JSONObject } from "../../../../src/common/jsonTypes";
import {
  DocumentEmbeddingsTransformer,
  VectorEmbedding,
} from "../../../../src/transformation/embeddings/embeddings";

export class TestEmbeddings extends DocumentEmbeddingsTransformer {
  constructor() {
    super(1536);
  }

  async embed(
    text: string,
    metadata?: JSONObject | undefined
  ): Promise<VectorEmbedding> {
    return {
      vector: [1, 2, 3],
      text,
      metadata: metadata ?? {},
      attributes: {},
    };
  }
}
