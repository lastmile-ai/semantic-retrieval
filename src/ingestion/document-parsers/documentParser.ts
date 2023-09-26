import { Attributable } from "../../common/base.js";
import { Document, RawDocument } from "../../document/document.js";

export interface DocumentParser extends Attributable {
  // The type of file that this parser can parse.
  mimeType: string;

  // TODO: saqadri - figure out how to parse a document from a stream (i.e. iteratively instead of all in one go)
  parse(rawDocument: RawDocument): Promise<Document>;

  /**
   * Parses the next fragment from the raw document.
   * @param rawDocument THe raw document to parse.
   * @param previousFragment The previous fragment in the document that was already parsed.
   * @param take The maximum number of fragments to parse.
   */
  parseNext(
    rawDocument: RawDocument,
    previousFragment?: DocumentFragment,
    take?: number,
  ): Promise<DocumentFragment>;

  /**
   * Converts the raw document to a string.
   * TODO: saqadri - figure out how to return a string stream for the document instead
   * so large documents aren't all loaded into memory.
   */
  toString(rawDocument: RawDocument): Promise<string>;
}
