import { DirectDocumentParser } from "./directDocumentParser";
import { DocumentParser } from "./documentParser";
import { TextDocumentParser } from "./textDocumentParser";

/**
 * A registry of document parsers, keyed by MIME type. By default, DirectDocumentParser
 * will be used unless a parser is registered for the MIME type of the document or
 * a different defaultParser (or null) is specified.
 */
export class ParserRegistry {
  // If no parser is found for a MIME type, this parser will be used, if it exists
  defaultParser: DocumentParser | undefined;
  parsers: Map<string, DocumentParser>;

  constructor(
    parsers?: Map<string, DocumentParser>,
    defaultParser?: DocumentParser | null
  ) {
    this.parsers =
      parsers ??
      new Map([
        ["text/plain", new TextDocumentParser()],
        // ['text/html', new HtmlDocumentParser()],
        // ['application/pdf', new PDFDocumentParser()],
      ]);

    if (defaultParser !== null) {
      this.defaultParser = defaultParser ?? new DirectDocumentParser();
    }
  }

  register(mimeType: string, parser: DocumentParser) {
    if (
      parser.mimeTypeRestriction &&
      !parser.mimeTypeRestriction.includes(mimeType)
    ) {
      throw new Error(
        `Parser ${parser.constructor.name} does not support MIME type ${mimeType}`
      );
    }
    this.parsers.set(mimeType, parser);
  }

  getParser(mimeType: string) {
    return this.parsers.get(mimeType) ?? this.defaultParser;
  }
}
