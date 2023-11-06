import {
  VectorDBBaseQuery,
  VectorDBTextQuery,
} from "../../data-store/vector-DBs/vectorDB";
import { VectorDBDocumentRetriever } from "../../retrieval/vector-DBs/vectorDBDocumentRetriever";
import {
  CompletionModel,
  ModelResponse,
} from "../completion-models/completionModel";
import {
  RAGCompletionGenerator,
  RAGCompletionGeneratorParams,
} from "./ragCompletionGenerator";

export interface VectorDBRAGCompletionGeneratorParams
  extends RAGCompletionGeneratorParams<VectorDBDocumentRetriever> {
  retrievalQuery?: VectorDBBaseQuery;
}

export class VectorDBRAGCompletionGenerator<
  M extends CompletionModel<ModelResponse<M>>,
> extends RAGCompletionGenerator<M, VectorDBDocumentRetriever> {
  async getRetrievalQuery(
    params: VectorDBRAGCompletionGeneratorParams
  ): Promise<VectorDBTextQuery> {
    const { prompt, retrievalQuery } = params;
    const text = typeof prompt === "string" ? prompt : await prompt.toString();
    const topK = retrievalQuery?.topK ?? 3;
    const query = {
      ...params.retrievalQuery,
      topK,
      text,
    };
    await this.callbackManager?.runCallbacks({
      name: "onGetRAGCompletionRetrievalQuery",
      params,
      query,
    });
    return query;
  }
}
