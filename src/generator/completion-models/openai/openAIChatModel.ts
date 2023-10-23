import getEnvVar from "../../../utils/getEnvVar";
import {
  ChatCompletion,
  ChatCompletionCreateParams,
  ChatCompletionCreateParamsBase,
  ChatCompletionMessageParam,
} from "openai/resources/chat/completions";
import { type ClientOptions as OpenAIClientOptions, OpenAI } from "openai";
import { CompletionModel, CompletionModelParams } from "../completionModel";

export type OpenAIChatModelParams =
  CompletionModelParams<ChatCompletionCreateParams>;

export interface OpenAIChatModelConfig extends OpenAIClientOptions {
  defaultModel?: string;
}

export class OpenAIChatModel extends CompletionModel<
  ChatCompletionCreateParams,
  ChatCompletion
> {
  private client: OpenAI;
  private defaultModel = "gpt-3.5-turbo";

  constructor(config?: OpenAIChatModelConfig) {
    super();

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

  async run(params: OpenAIChatModelParams): Promise<ChatCompletion> {
    const completionParams = params.completionParams;
    const model = params.model ?? completionParams?.model ?? this.defaultModel;

    const refinedCompletionParams: ChatCompletionCreateParamsBase = {
      ...completionParams,
      model,
      messages: await this.constructMessages(params),
    };

    if (completionParams?.stream) {
      throw new Error("Streamed completions not implemented yet");
    } else {
      return (await this.client.chat.completions.create(
        refinedCompletionParams
      )) as ChatCompletion;
    }
  }
}
