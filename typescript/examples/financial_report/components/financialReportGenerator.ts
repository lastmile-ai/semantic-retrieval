import { AccessPassport } from "../../../src/access-control/accessPassport";
import { CompletionModelParams } from "../../../src/generator/completion-models/completionModel";
import { LLMCompletionGenerator } from "../../../src/generator/completionGenerator";
import { FinancialReportDocumentRetriever } from "./financialReportDocumentRetriever";
import {
  AIConfigCompletion,
  AIConfigPromptParams,
} from "../../../src/generator/completion-models/aiconfig/aiconfigCompletionModel";
import { CallbackManager } from "../../../src/utils/callbacks";
import { Output } from "aiconfig";
import { JSONObject } from "aiconfig/dist/common";
import * as path from "path";

interface FinancialReportGeneratorParams extends CompletionModelParams {
  accessPassport: AccessPassport;
  retriever: FinancialReportDocumentRetriever;
}

export class FinancialReportGenerator extends LLMCompletionGenerator {
  model: AIConfigCompletion;

  constructor(callbackManager?: CallbackManager) {
    const model = new AIConfigCompletion(
      path.join(__dirname, "report.aiconfig.json"),
      callbackManager
    );
    super(model, callbackManager);
    this.model = model;
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

    return processResponse(response.data);
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
