import {
  CompletionModel,
  CompletionModelParams,
} from "../../../src/generator/completion-models/completionModel";
import { CallbackManager } from "../../../src/utils/callbacks";

export type TestCompletionRequestParams = { prompt: string };
export type TestCompletionResponse = { completion: string };

export type TestCompletionModelParams =
  CompletionModelParams<TestCompletionRequestParams>;

export class TestCompletionModel extends CompletionModel<
  TestCompletionRequestParams,
  TestCompletionResponse
> {
  constructor(callbackManager?: CallbackManager) {
    super(callbackManager);
  }

  async run(
    params: TestCompletionModelParams
  ): Promise<TestCompletionResponse> {
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

    return response;
  }
}
