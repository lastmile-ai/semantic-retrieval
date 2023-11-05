import {
  VectorDBBaseQuery,
  VectorDBTextQuery,
} from "../../data-store/vector-DBs/vectorDB";
import { VectorDBDocumentRetriever } from "../../retrieval/vector-DBs/vectorDBDocumentRetriever";
import {
  CompletionModel,
  ModelParams,
  ModelResponse,
} from "../completion-models/completionModel";
import {
  RAGCompletionGenerator,
  RAGCompletionGeneratorParams,
} from "./ragCompletionGenerator";

export interface VectorDBRAGCompletionGeneratorParams<
  M extends CompletionModel<ModelParams<M>, ModelResponse<M>>,
> extends RAGCompletionGeneratorParams<
    ModelParams<M>,
    VectorDBDocumentRetriever
  > {
  retrievalQuery?: VectorDBBaseQuery;
}

export class VectorDBRAGCompletionGenerator<
  M extends CompletionModel<ModelParams<M>, ModelResponse<M>>,
> extends RAGCompletionGenerator<
  M,
  VectorDBRAGCompletionGeneratorParams<M>,
  VectorDBDocumentRetriever
> {
  async getRetrievalQuery(
    params: VectorDBRAGCompletionGeneratorParams<M>
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
