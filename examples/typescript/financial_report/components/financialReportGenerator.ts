import { AccessPassport } from "../../../../src/access-control/accessPassport";
import { CompletionModelParams } from "../../../../src/generator/completion-models/completionModel";
import { LLMCompletionGenerator } from "../../../../src/generator/completionGenerator";
import { FinancialReportDocumentRetriever } from "./financialReportDocumentRetriever";
import { PromptTemplate } from "../../../../src/prompts/prompt-templates/promptTemplate";
import { OpenAIChatModel } from "../../../../src/generator/completion-models/openai/openAIChatModel";
import { ChatCompletion, ChatCompletionCreateParams } from "openai/resources";
import { CallbackManager } from "../../../../src/utils/callbacks";

interface FinancialReportGeneratorParams
  extends CompletionModelParams<ChatCompletionCreateParams> {
  accessPassport: AccessPassport;
  retriever: FinancialReportDocumentRetriever;
  structure: string;
}

const PROMPT_TEMPLATE = "STRUCTURE: {{structure}}; CONTEXT: {{companyDetails}}";

export class FinancialReportGenerator<P, R> extends LLMCompletionGenerator<
  ChatCompletionCreateParams,
  ChatCompletion,
  FinancialReportGeneratorParams,
  string
> {
  constructor(callbackManager?: CallbackManager) {
    super(new OpenAIChatModel(), callbackManager);
  }

  async run(params: FinancialReportGeneratorParams): Promise<string> {
    const { accessPassport, prompt, retriever, ...modelParams } = params;

    const detailsPrompt =
      typeof prompt === "string" ? prompt : await prompt.toString();

    const companyDetails = await retriever.retrieveData({
      accessPassport,
      query: detailsPrompt,
    });

    const completionPrompt = new PromptTemplate(PROMPT_TEMPLATE, {
      //topic: detailsPrompt,
      companyDetails: JSON.stringify(companyDetails, null, 2),
      structure: params.structure,
    });

    const response = await this.model.run({
      ...modelParams,
      completionParams: {
        messages: [
          {
            content:
              "INSTRUCTIONS: You are a helpful assistant. Rearrange the context to answer the question. " +
              "Output your response following the requested structure. Do not include any words that do not appear in the context.",
            role: "system",
          },
        ],
        model: "gpt-3.5-turbo-16k",
        stream: false,
      },
      prompt: completionPrompt,
    });

    await this.callbackManager?.runCallbacks({
      name: "onRunCompletionGeneration",
      params,
      response,
    });

    return response.choices[0].message.content ?? "";
  }
}
