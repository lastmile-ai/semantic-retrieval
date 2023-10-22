import {
  VectorDBBaseQuery,
  VectorDBQuery,
  VectorDBTextQuery,
} from "../../data-store/vector-DBs/vectorDB";
import { VectorDBDocumentRetriever } from "../../retrieval/vector-DBs/vectorDBDocumentRetriever";
import {
  RAGCompletionGenerator,
  RAGCompletionGeneratorParams,
} from "./ragCompletionGenerator";

export interface VectorDBRAGCompletionGeneratorParams<P>
  extends RAGCompletionGeneratorParams<P, VectorDBQuery> {
  retriever: VectorDBDocumentRetriever;
  retrievalQuery?: VectorDBBaseQuery;
}

export class VectorDBRAGCompletionGenerator<
  P,
  R,
> extends RAGCompletionGenerator<
  P,
  VectorDBTextQuery,
  R,
  VectorDBRAGCompletionGeneratorParams<P>
> {
  async getRetrievalQuery(
    params: VectorDBRAGCompletionGeneratorParams<P>
  ): Promise<VectorDBTextQuery> {
    const prompt = await params.prompt.toString();
    const topK = params.retrievalQuery?.topK ?? 3;
    return {
      ...params.retrievalQuery,
      topK,
      text: prompt,
    } as VectorDBTextQuery;
  }
}
