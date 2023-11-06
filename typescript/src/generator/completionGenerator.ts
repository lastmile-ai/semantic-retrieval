import { CallbackManager, Traceable } from "../utils/callbacks";
import { CompletionModel } from "./completion-models/completionModel";

/**
 * A CompletionGenerator generates some response resulting from completion requests to
 * one or more CompletionModels.
 */
export abstract class CompletionGenerator implements Traceable {
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
  abstract run(params: unknown): Promise<unknown>;
}

/**
 * LLM Completion Generators are used for generating some completion response from an
 * LLM Completion Model based on input parameters and any applicable internal logic.
 */
export abstract class LLMCompletionGenerator<
    M extends CompletionModel<MR>,
    MR = M extends CompletionModel<infer R> ? R : never,
  >
  extends CompletionGenerator
  implements Traceable
{
  model: M;

  constructor(model: M, callbackManager?: CallbackManager) {
    super(callbackManager);
    this.model = model;
  }

  abstract run(params: unknown): Promise<unknown>;
}
