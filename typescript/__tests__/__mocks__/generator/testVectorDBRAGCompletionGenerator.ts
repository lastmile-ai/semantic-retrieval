import { VectorDBTextQuery } from "../../../src/data-store/vector-DBs/vectorDB";
import {
  VectorDBRAGCompletionGenerator,
  VectorDBRAGCompletionGeneratorParams,
} from "../../../src/generator/retrieval-augmented-generation/vectorDBRAGCompletionGenerator";
import { TestCompletionModel } from "./testCompletionModel";

export class TestVectorDBRAGCompletionGenerator extends VectorDBRAGCompletionGenerator<TestCompletionModel> {
  async getRetrievalQuery(
    params: VectorDBRAGCompletionGeneratorParams
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
