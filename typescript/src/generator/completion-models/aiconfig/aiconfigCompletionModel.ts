import {
  CompletionModel,
  CompletionModelParams,
  CompletionModelResponse,
} from "../completionModel";
import { CallbackManager } from "../../../utils/callbacks";

import { AIConfigRuntime, Output } from "aiconfig";
import type { JSONObject } from "aiconfig/dist/common";

export interface AIConfigPromptParams extends CompletionModelParams {
  params: JSONObject;
}

export interface AIConfigCompletionResponse
  extends CompletionModelResponse<Output | Output[]> {}

export class AIConfigCompletion extends CompletionModel {
  private aiConfig: AIConfigRuntime;

  constructor(aiConfigFilePath: string, callbackManager?: CallbackManager) {
    super(callbackManager);

    this.aiConfig = AIConfigRuntime.load(aiConfigFilePath);
  }

  async run(input: AIConfigPromptParams): Promise<AIConfigCompletionResponse> {
    const { prompt, params } = input;

    // Assuming prompt is prompt name for AIConfig
    if (typeof prompt !== "string") {
      throw new Error("Unexpected prompt type, expecting prompt name");
    }

    const resolvedPrompt = await this.aiConfig.resolve(prompt, params);
    await this.callbackManager?.runCallbacks({
      name: "onRunCompletionRequest",
      params: {
        prompt,
        ...resolvedPrompt,
      },
    });

    const result = await this.aiConfig.run(prompt, params, {
      stream: true,
      callbacks: {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        streamCallback: (data: any, _accumulatedData: any, _index: any) => {
          process.stdout.write(data?.content || "\n");
        },
      },
    });

    if (!result) {
      throw new Error("Unexpected -- inference returned no result");
    }

    await this.callbackManager?.runCallbacks({
      name: "onRunCompletionResponse",
      params: {
        prompt,
        ...resolvedPrompt,
      },
      response: result,
    });

    return { data: result };
  }
}
