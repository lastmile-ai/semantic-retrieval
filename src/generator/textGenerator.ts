import { GeneratorParams } from "./generator";
import { Document } from "../document/document";

/**
 * Simple abstract class for generating text from a prompt and retrieved Documents.
 */
export abstract class TextGenerator {
  abstract run(params: GeneratorParams<Document[]>): Promise<string>;
}
