import {
  VectorDBBaseQuery,
  VectorDBTextQuery,
} from "../../data-store/vector-DBs/vectorDB";
import { VectorDBDocumentRetriever } from "../../retrieval/vector-DBs/vectorDBDocumentRetriever";
import {
  RAGCompletionGenerator,
  RAGCompletionGeneratorParams,
} from "./ragCompletionGenerator";

export interface VectorDbRAGCompletionGeneratorParams
  extends RAGCompletionGeneratorParams<VectorDBDocumentRetriever> {
  retrievalQuery?: VectorDBBaseQuery;
}

export class VectorDbRAGCompletionGenerator extends RAGCompletionGenerator<VectorDBDocumentRetriever> {
  async getRetrievalQuery(
    params: VectorDbRAGCompletionGeneratorParams
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
