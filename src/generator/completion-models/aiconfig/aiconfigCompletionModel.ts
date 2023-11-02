import { CompletionModel, CompletionModelParams } from "../completionModel";
import { CallbackManager } from "../../../utils/callbacks";

import { AIConfigRuntime, Output } from "aiconfig";
import type { JSONObject } from "aiconfig/dist/common";

export interface AIConfigPromptParams extends CompletionModelParams<never> {
  params: JSONObject;
}

export class AIConfigCompletion extends CompletionModel<
  never,
  Output | Output[]
> {
  private aiConfig: AIConfigRuntime;

  constructor(aiConfigFilePath: string, callbackManager?: CallbackManager) {
    super(callbackManager);

    this.aiConfig = AIConfigRuntime.load(aiConfigFilePath);
  }

  async run(input: AIConfigPromptParams): Promise<Output | Output[]> {
    const { prompt, params } = input;

    // Assuming prompt is prompt name for AIConfig
    if (typeof prompt !== "string") {
      throw new Error("Unexpected prompt type, expecting prompt name");
    }

    const resolvedPrompt = await this.aiConfig.resolve(prompt, params);
    console.log(
      `About to execute prompt: ${JSON.stringify(resolvedPrompt, null, 2)}`
    );

    const result = await this.aiConfig.run(prompt, params, {
      stream: true,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      streamCallback: (data: any, _accumulatedData: any, _index: any) => {
        console.log(data?.content);
        process.stdout.write(data?.content || "\n");
      },
    });

    if (!result) {
      throw new Error("Unexpected -- inference returned no result");
    }

    return result;
  }
}
