import {
  CompletionModel,
  CompletionModelParams,
  CompletionModelResponse,
} from "../../../src/generator/completion-models/completionModel";
import { CallbackManager } from "../../../src/utils/callbacks";

export interface TestCompletionModelParams
  extends CompletionModelParams<{ prompt: string }> {}

export interface TestCompletionModelResponse
  extends CompletionModelResponse<{ completion: string }> {}

export class TestCompletionModel extends CompletionModel {
  constructor(callbackManager?: CallbackManager) {
    super(callbackManager);
  }

  async run(
    params: TestCompletionModelParams
  ): Promise<TestCompletionModelResponse> {
    await this.callbackManager?.runCallbacks({
      name: "onRunCompletionRequest",
      params,
    });

    const response = {
      completion: "test completion",
    };

    await this.callbackManager?.runCallbacks({
      name: "onRunCompletionResponse",
      params,
      response,
    });

    return { data: response };
  }
}
