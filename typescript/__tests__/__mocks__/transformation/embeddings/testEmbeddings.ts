import { JSONObject } from "../../../../src/common/jsonTypes";
import {
  DocumentEmbeddingsTransformer,
  VectorEmbedding,
} from "../../../../src/transformation/embeddings/embeddings";

export const TEST_VECTOR = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

export class TestEmbeddings extends DocumentEmbeddingsTransformer {
  constructor() {
    super(1536);
  }

  async embed(
    text: string,
    metadata?: JSONObject | undefined
  ): Promise<VectorEmbedding> {
    return {
      vector: TEST_VECTOR,
      text,
      metadata: metadata ?? {},
      attributes: {},
    };
  }
}
