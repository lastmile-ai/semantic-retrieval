import { RawDocument, Document } from "../../document/document";
import { DocumentParser } from "./documentParser";

/**
 * A basic DocumentParser implementation for text/plain documents.
 */
export class TextDocumentParser implements DocumentParser {
  mimeType = "text/plain";

  attributes = {};
  metadata = {};

  parse(_rawDocument: RawDocument): Promise<Document> {
    throw new Error("Method not implemented.");
  }

  parseNext(
    _rawDocument: RawDocument,
    _previousFragment?: DocumentFragment | undefined,
    _take?: number | undefined,
  ): Promise<DocumentFragment> {
    throw new Error("Method not implemented.");
  }

  toString(_rawDocument: RawDocument): Promise<string> {
    throw new Error("Method not implemented.");
  }
}
