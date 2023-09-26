import { ParserRegistry } from "../ingestion/document-parsers/parserRegistry.js";

export interface Context {
  parsers: ParserRegistry;
}
