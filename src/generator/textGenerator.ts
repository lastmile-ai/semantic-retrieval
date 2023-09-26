import { GeneratorParams } from "./generator.js";
import { Document } from "../document/document.js";

/**
 * Simple abstract class for generating text from a prompt and retrieved Documents.
 */
export abstract class TextGenerator {
  protected abstract run(params: GeneratorParams<Document[]>): Promise<string>;
}
