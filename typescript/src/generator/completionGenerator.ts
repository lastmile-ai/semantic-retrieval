import { CallbackManager, Traceable } from "../utils/callbacks";
import { CompletionModel } from "./completion-models/completionModel";

/**
 * A CompletionGenerator generates some response resulting from completion requests to
 * one or more CompletionModels.
 */
export abstract class CompletionGenerator<P = unknown, R = unknown>
  implements Traceable
{
  callbackManager?: CallbackManager;

  constructor(callbackManager?: CallbackManager) {
    this.callbackManager = callbackManager;
  }

  /**
   * Perform completion generation using the given parameters and return response
   * data
   * @param params The parameters to use for generating the completion
   * @returns A promise that resolves to the generated completion data
   */
  abstract run(params: P): Promise<R>;
}

/**
 * LLM Completion Generators are used for generating some completion response from an
 * LLM Completion Model based on input parameters and any applicable internal logic.
 */
export abstract class LLMCompletionGenerator<P = unknown, R = unknown>
  extends CompletionGenerator<P, R>
  implements Traceable
{
  model: CompletionModel;

  constructor(model: CompletionModel, callbackManager?: CallbackManager) {
    super(callbackManager);
    this.model = model;
  }

  abstract run(params: P): Promise<R>;
}
