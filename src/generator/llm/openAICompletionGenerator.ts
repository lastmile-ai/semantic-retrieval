import { JSONObject } from "../../common/jsonTypes";
import { GeneratorParams } from "../generator";
import { Document } from "../../document/document";
import { TextGenerator } from "../textGenerator";

export type OpenAICompletionGeneratorParams = GeneratorParams<Document[]> & {
  completionParams?: JSONObject; // TODO: Add openai API completion params type
};

// TODO: Implement this for different openai completion models
export class OpenAICompletionGenerator extends TextGenerator {
  constructor() {
    super();
    // TODO: Initialize OpenAI API client
  }

  run(_params: OpenAICompletionGeneratorParams): Promise<string> {
    throw new Error("Method not implemented.");
  }
}
