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
}

const PROMPT_TEMPLATE = `For each group of {company: <company>, profile: <profile>, details: <details>} in DETAILS, generate a brief paragraph of the form: 
** <company> **:
Info: Exact content of <profile>
{{topic}}: summary of <details> with respect to {{topic}}

DETAILS: {{companyDetails}}`;

export class FinancialReportGenerator<P, R> extends LLMCompletionGenerator<
  ChatCompletionCreateParams,
  ChatCompletion,
  FinancialReportGeneratorParams,
  string
> {
  constructor(callbackManager?: CallbackManager) {
    super(new OpenAIChatModel({ callbackManager }), callbackManager);
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
      topic: detailsPrompt,
      companyDetails: JSON.stringify(companyDetails, null, 2),
    });

    const response = await this.model.run({
      completionParams: {
        messages: [],
        model: "gpt-3.5-turbo-16k",
        stream: false,
      },
      ...modelParams,
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
