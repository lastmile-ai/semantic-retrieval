import { VectorDBTextQuery } from "../../../src/data-store/vector-DBs/vectorDB";
import {
  VectorDbRAGCompletionGenerator,
  VectorDbRAGCompletionGeneratorParams,
} from "../../../src/generator/retrieval-augmented-generation/vectorDbRAGCompletionGenerator";

export class TestVectorDbRAGCompletionGenerator extends VectorDbRAGCompletionGenerator {
  async getRetrievalQuery(
    params: VectorDbRAGCompletionGeneratorParams
  ): Promise<VectorDBTextQuery> {
    const query = {
      topK: 1,
      text: "test",
    };
    await this.callbackManager?.runCallbacks({
      name: "onGetRAGCompletionRetrievalQuery",
      params,
      query,
    });
    return query;
  }
}
