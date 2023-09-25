import { JSONObject } from "../../common/jsonTypes";
import { BaseGenerator, GeneratorParams } from "../generator";
import { Document } from "../../document/document";

export type OpenAICompletionGeneratorParams<T> = GeneratorParams<T> & {
  completionParams?: JSONObject; // TODO: Add openai API completion params type
};

// TODO: Think on this class structure some more. As-is, we'll need custom generators for 
// each retrieved type and post-processed response type
export class OpenAICompletionGenerator extends BaseGenerator<
  Document[],
  string
> {
  constructor() {
    super();
    // TODO: Initialize OpenAI API client
  }

  run(_params: OpenAICompletionGeneratorParams<Document[]>): Promise<string> {
    throw new Error("Method not implemented.");
  }
}
