import { DirectDocumentParser } from "./directDocumentParser";
import { DocumentParser, DocumentParserConfig } from "./documentParser";

export interface ParserRegistryConfig extends DocumentParserConfig {
  defaultParser?: DocumentParser | null;
  parsers?: Map<string, DocumentParser>;
}

/**
 * A registry of document parsers, keyed by MIME type. By default, DirectDocumentParser
 * will be used unless a parser is registered for the MIME type of the document or
 * a different defaultParser (or null) is specified.
 */
export class ParserRegistry {
  // If no parser is found for a MIME type, this parser will be used, if it exists
  defaultParser: DocumentParser | undefined;
  parsers: Map<string, DocumentParser>;

  constructor(config: ParserRegistryConfig) {
    this.parsers =
      config.parsers ??
      new Map([
        // ["text/plain", new TextDocumentParser(config)],
        // ['text/html', new HtmlDocumentParser(config)],
        // ['application/pdf', new PDFDocumentParser(config)],
      ]);

    if (config.defaultParser !== null) {
      this.defaultParser =
        config.defaultParser ?? new DirectDocumentParser(config);
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
