import { AccessPassport } from "../../../../src/access-control/accessPassport";
import { CompletionModelParams } from "../../../../src/generator/completion-models/completionModel";
import { LLMCompletionGenerator } from "../../../../src/generator/completionGenerator";
import { FinancialReportDocumentRetriever } from "./financialReportDocumentRetriever";
import {
  AIConfigCompletion,
  AIConfigPromptParams,
} from "../../../../src/generator/completion-models/aiconfig/aiconfigCompletionModel";
import { CallbackManager } from "../../../../src/utils/callbacks";
import { Output } from "aiconfig";
import { JSONObject } from "aiconfig/dist/common";
import * as path from "path";

interface FinancialReportGeneratorParams extends CompletionModelParams<never> {
  accessPassport: AccessPassport;
  retriever: FinancialReportDocumentRetriever;
}

export class FinancialReportGenerator<P, R> extends LLMCompletionGenerator<
  never,
  Output | Output[],
  FinancialReportGeneratorParams,
  string
> {
  constructor(callbackManager?: CallbackManager) {
    super(
      new AIConfigCompletion(
        path.join(__dirname, "report.aiconfig.json"),
        callbackManager
      ),
      callbackManager
    );
  }

  async run(params: FinancialReportGeneratorParams): Promise<string> {
    const { accessPassport, prompt, retriever } = params;

    const detailsPrompt =
      typeof prompt === "string" ? prompt : await prompt.toString();

    const companyDetails = await retriever.retrieveData({
      accessPassport,
      query: detailsPrompt,
    });

    const response = await this.model.run({
      prompt: "topicalSummary",
      params: {
        topic: detailsPrompt,
        companyDetails: JSON.stringify(companyDetails, null, 2),
      },
    } as AIConfigPromptParams);

    await this.callbackManager?.runCallbacks({
      name: "onRunCompletionGeneration",
      params,
      response,
    });

    return processResponse(response);
  }
}

function processResponse(response: Output | Output[]) {
  let responseText = "";
  let output: Output;
  if (Array.isArray(response)) {
    output = response[0];
  } else {
    output = response;
  }

  if (output.output_type === "execute_result") {
    responseText = ((output.data as JSONObject)?.content as string) ?? "";
  } else {
    throw new Error(
      `Encountered error during inference: ${output.ename} - ${output.evalue}`
    );
  }

  return responseText;
}
