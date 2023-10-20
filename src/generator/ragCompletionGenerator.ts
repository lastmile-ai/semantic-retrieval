import { AccessPassport } from "../access-control/accessPassport";
import { PromptTemplate } from "../prompts/prompt_templates/promptTemplate";
import { DocumentRetriever } from "../retrieval/documentRetriever";
import {
  LLMCompletionGeneratorParams,
  LLMCompletionGenerator,
} from "./completionGenerator";

export interface RAGCompletionGeneratorParams<P, Q>
  extends LLMCompletionGeneratorParams<P> {
  accessPassport: AccessPassport;
  retriever: DocumentRetriever<Q>;
  ragPromptTemplate?: PromptTemplate;
}

export const DEFAULT_RAG_TEMPLATE =
  "Answer the question based on the context below.\n\nContext:\n\n{{context}}\n\nQuestion: {{prompt}}\n\nAnswer:";

/**
 * A basic RAG completion generator that uses a retriever to retrieve documents which can
 * be leveraged for modifying the prompt prior to completion generation by the model
 */
export class RAGCompletionGenerator<P, Q, R> extends LLMCompletionGenerator<
  P,
  R
> {
  /**
   * Performs completion generation using the given parameters and returns the generated
   * response data
   * @param params The parameters to use for generating the completion
   * @returns A promise that resolves to the generated completion data
   */
  async run(params: RAGCompletionGeneratorParams<P, Q>): Promise<R> {
    const { accessPassport, prompt, retriever, ...modelParams } = params;

    const queryPrompt = await prompt.toString();

    const contextDocs = await retriever.retrieveData({
      accessPassport,
      query: queryPrompt,
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

    return await this.model.run({ ...modelParams, prompt: ragPromptTemplate });
  }
}
