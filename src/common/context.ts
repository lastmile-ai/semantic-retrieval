import { ParserRegistry } from "../ingestion/document-parsers/parserRegistry";

export interface Context {
  parsers: ParserRegistry;
}
