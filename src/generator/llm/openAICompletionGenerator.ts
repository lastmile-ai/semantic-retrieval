import { JSONObject } from "../../common/jsonTypes.js";
import { GeneratorParams } from "../generator.js";
import { Document } from "../../document/document.js";
import { TextGenerator } from "../textGenerator.js";

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
