import { CallbackManager, Traceable } from "../utils/callbacks";
import {
  CompletionModel,
  CompletionModelParams,
} from "./completion-models/completionModel";

export interface LLMCompletionGeneratorParams<P>
  extends CompletionModelParams<P> {}

/**
 * LLM Completion Generators are used for generating some completion response from an LLM
 * based on a prompt and any applicable internal logic. Extend this base class to add
 * additional logic.
 */
export abstract class LLMCompletionGenerator<P, R> implements Traceable {
  model: CompletionModel<P, R>;
  callbackManager?: CallbackManager;

  constructor(model: CompletionModel<P, R>, callbackManager?: CallbackManager) {
    this.model = model;
    this.callbackManager = callbackManager;
  }

  /**
   * Perform completion generation using the given parameters and return the generated
   * response data
   * @param params The parameters to use for generating the completion
   * @returns A promise that resolves to the generated completion data
   */
  abstract run(params: LLMCompletionGeneratorParams<P>): Promise<R>;
}
