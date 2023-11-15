import getEnvVar from "../../../utils/getEnvVar";
import {
  ChatCompletion,
  ChatCompletionCreateParams,
  ChatCompletionCreateParamsBase,
  ChatCompletionMessageParam,
} from "openai/resources/chat/completions";
import { type ClientOptions as OpenAIClientOptions, OpenAI } from "openai";
import {
  CompletionModel,
  CompletionModelParams,
  CompletionModelResponse,
} from "../completionModel";
import { CallbackManager } from "../../../utils/callbacks";

export interface OpenAIChatModelParams
  extends CompletionModelParams<ChatCompletionCreateParams> {}

export interface OpenAIChatModelResponse
  extends CompletionModelResponse<ChatCompletion> {}

export interface OpenAIChatModelConfig extends OpenAIClientOptions {
  callbackManager?: CallbackManager;
  defaultModel?: string;
}

export class OpenAIChatModel extends CompletionModel {
  private client: OpenAI;
  private defaultModel = "gpt-3.5-turbo";

  constructor(config?: OpenAIChatModelConfig) {
    super(config?.callbackManager);

    const apiKey = config?.apiKey ?? getEnvVar("OPENAI_API_KEY");
    if (!apiKey) {
      throw new Error(
        "No OpenAI API key found for OpenAIChatCompletionGenerator"
      );
    }

    this.defaultModel = config?.defaultModel ?? this.defaultModel;

    this.client = new OpenAI({ ...config, apiKey });
  }

  /**
   * Construct the completion request messages, adding the prompt from params
   * as the latest "user" message
   * @param params OpenAIChatModelParams The params to use for constructing messages
   * @returns ChatCompletionMessageParam[] Constructed messages including the prompt
   */
  private async constructMessages(
    params: OpenAIChatModelParams
  ): Promise<ChatCompletionMessageParam[]> {
    const messages: ChatCompletionMessageParam[] =
      params.completionParams?.messages ?? [];

    const content = await params.prompt.toString();

    messages.push({ content, role: "user" });
    return messages;
  }

  async run(params: OpenAIChatModelParams): Promise<OpenAIChatModelResponse> {
    const completionParams = params.completionParams;
    const model = params.model ?? completionParams?.model ?? this.defaultModel;

    const refinedCompletionParams: ChatCompletionCreateParamsBase = {
      ...completionParams,
      model,
      messages: await this.constructMessages(params),
    };

    await this.callbackManager?.runCallbacks({
      name: "onRunCompletionRequest",
      params: {
        ...params,
        completionParams: refinedCompletionParams,
      },
    });

    if (completionParams?.stream) {
      throw new Error("Streamed completions not implemented yet");
    } else {
      const response = (await this.client.chat.completions.create(
        refinedCompletionParams
      )) as ChatCompletion;

      await this.callbackManager?.runCallbacks({
        name: "onRunCompletionResponse",
        params,
        response,
      });

      return { data: response };
    }
  }
}
