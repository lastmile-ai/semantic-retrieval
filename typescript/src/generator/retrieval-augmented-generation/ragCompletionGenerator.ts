import { AccessPassport } from "../../access-control/accessPassport";
import { PromptTemplate } from "../../prompts/prompt-templates/promptTemplate";
import { Document } from "../../document/document";
import { DocumentRetriever } from "../../retrieval/documentRetriever";
import { LLMCompletionGenerator } from "../completionGenerator";
import {
  CompletionModelParams,
  CompletionModelResponse,
} from "../completion-models/completionModel";
import { RetrieverQuery } from "../../retrieval/retriever";

export interface RAGCompletionGeneratorParams<
  R extends DocumentRetriever<Document[]>,
  P = unknown,
> extends CompletionModelParams<P> {
  retriever: R;
  accessPassport?: AccessPassport;
  ragPromptTemplate?: PromptTemplate;
}

export const DEFAULT_RAG_TEMPLATE =
  "Answer the question based on the context below.\n\nContext:\n\n{{context}}\n\nQuestion: {{prompt}}\n\nAnswer:";

/**
 * A basic RAG completion generator that uses a retriever to retrieve documents which can
 * be leveraged for modifying the prompt prior to completion generation by the model
 */
export abstract class RAGCompletionGenerator<
  R extends DocumentRetriever<Document[]>,
  P extends RAGCompletionGeneratorParams<R> = RAGCompletionGeneratorParams<R>,
> extends LLMCompletionGenerator {
  /**
   * Construct the query for the underlying retriever using the given parameters
   * @param params The parameters to use for constructing the query
   * @returns A promise that resolves to the query in valid format for the retriever
   */
  abstract getRetrievalQuery(params: P): Promise<RetrieverQuery<R>>;

  /**
   * Performs completion generation using the given parameters and returns the generated
   * response data
   * @param params The parameters to use for generating the completion
   * @returns A promise that resolves to the generated completion data
   */
  async run(params: P): Promise<CompletionModelResponse> {
    const { accessPassport, prompt, retriever, ...modelParams } = params;

    const queryPrompt =
      typeof prompt === "string" ? prompt : await prompt.toString();

    const contextDocs = await retriever.retrieveData({
      accessPassport,
      query: await this.getRetrievalQuery(params),
    });

    const contextChunksPromises = [];
    for (const doc of contextDocs) {
      for (const fragment of doc.fragments) {
        contextChunksPromises.push(fragment.getContent());
      }
    }

    const context = (await Promise.all(contextChunksPromises)).join("\n");
    const ragPromptTemplate =
      params.ragPromptTemplate ?? new PromptTemplate(DEFAULT_RAG_TEMPLATE);

    ragPromptTemplate.setParameters({
      prompt: queryPrompt,
      context,
    });

    const response = await this.model.run({
      ...modelParams,
      prompt: ragPromptTemplate,
    });

    await this.callbackManager?.runCallbacks({
      name: "onRunCompletionGeneration",
      params,
      response,
    });

    return response;
  }
}
