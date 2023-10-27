import { AccessPassport } from "../../access-control/accessPassport";
import { PromptTemplate } from "../../prompts/prompt-templates/promptTemplate";
import { Document } from "../../document/document";
import { DocumentRetriever } from "../../retrieval/documentRetriever";
import {
  LLMCompletionGeneratorParams,
  LLMCompletionGenerator,
} from "../completionGenerator";

export interface RAGCompletionGeneratorParams<P, Q>
  extends LLMCompletionGeneratorParams<P> {
  retriever: DocumentRetriever<Document[], Q>;
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
  P,
  Q,
  R,
  T extends RAGCompletionGeneratorParams<P, Q>,
> extends LLMCompletionGenerator<P, R> {
  /**
   * Construct the query for the underlying retriever using the given parameters
   * @param params The parameters to use for constructing the query
   * @returns A promise that resolves to the query in valid format for the retriever
   */
  abstract getRetrievalQuery(params: T): Promise<Q>;

  /**
   * Performs completion generation using the given parameters and returns the generated
   * response data
   * @param params The parameters to use for generating the completion
   * @returns A promise that resolves to the generated completion data
   */
  async run(params: T): Promise<R> {
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
