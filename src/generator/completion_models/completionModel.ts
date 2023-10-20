import { IPrompt } from "../../prompts/prompt";

export interface CompletionModelParams<P> {
  prompt: IPrompt;
  model?: string;
  completionParams?: P;
}

/**
 * A simple class for interacting with different LLM completion models. CompletionModels
 * are leveraged by CompletionGenerators to generate completions from prompts.
 */
export abstract class CompletionModel<P, R> {
  abstract run(params: CompletionModelParams<P>): Promise<R>;
}
