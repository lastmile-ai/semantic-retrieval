import { DocumentParser } from "./documentParser";

/**
 * A registry of document parsers, keyed by MIME type.
 */
export class ParserRegistry {
  // TODO: saqadri = instantiate this map with default parsers for various MIME types.
  parsers: Map<string, DocumentParser> = new Map(
    [
      ['text/plain', new TextDocumentParser()],
      // ['text/html', new HtmlDocumentParser()],
      // ['application/pdf', new PDFDocumentParser()],
    ]
  );

  constructor() {}

  register(parser: DocumentParser) {
    this.parsers.set(parser.mimeType, parser);
  }

  getParser(mimeType: string) {
    return this.parsers.get(mimeType);
  }
}
