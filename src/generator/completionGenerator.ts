import {
  CompletionModel,
  CompletionModelParams,
} from "./completion_models/completionModel";

export interface LLMCompletionGeneratorParams<P>
  extends CompletionModelParams<P> {}

/**
 * LLM Completion Generators are used for generating some completion response from an LLM
 * based on a prompt and any applicable internal logic. Extend this base class to add
 * additional logic.
 */
export class LLMCompletionGenerator<P, R> {
  model: CompletionModel<P, R>;

  constructor(model: CompletionModel<P, R>) {
    this.model = model;
  }

  /**
   * Perform completion generation using the given parameters and return the generated
   * response data
   * @param params The parameters to use for generating the completion
   * @returns A promise that resolves to the generated completion data
   */
  async run(params: LLMCompletionGeneratorParams<P>): Promise<R> {
    return await this.model.run(params);
  }
}
