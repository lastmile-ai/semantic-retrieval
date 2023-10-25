import { IPrompt } from "../../prompts/prompt";
import { CallbackManager, Traceable } from "../../utils/callbacks";

export interface CompletionModelParams<P> {
  prompt: string | IPrompt;
  model?: string;
  completionParams?: P;
}

/**
 * A simple class for interacting with different LLM completion models. CompletionModels
 * are leveraged by CompletionGenerators to generate completions from prompts.
 */
export abstract class CompletionModel<P, R> implements Traceable {
  callbackManager?: CallbackManager;

  constructor(callbackManager?: CallbackManager) {
    this.callbackManager = callbackManager;
  }

  abstract run(params: CompletionModelParams<P>): Promise<R>;
}
