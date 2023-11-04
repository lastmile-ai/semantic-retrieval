import { Attributable } from "../../common/base";
import { JSONObject } from "../../common/jsonTypes";
import {
  RawDocument,
  DocumentFragment,
  IngestedDocument,
} from "../../document/document";

import { CallbackManager, Traceable } from "../../utils/callbacks";

export interface DocumentParser extends Attributable, Traceable {
  // If applicable, restrict the DocumentParser to only parse documents with the specified MIME types.
  mimeTypeRestriction?: string[];

  // TODO: saqadri - figure out how to parse a document from a stream (i.e. iteratively instead of all in one go)
  parse(rawDocument: RawDocument): Promise<IngestedDocument>;

  /**
   * Parses the next fragment from the raw document.
   * @param rawDocument THe raw document to parse.
   * @param previousFragment The previous fragment in the document that was already parsed.
   * @param take The maximum number of fragments to parse.
   */
  parseNext(
    rawDocument: RawDocument,
    previousFragment?: DocumentFragment,
    take?: number
  ): Promise<DocumentFragment>;

  /**
   * Converts the raw document to a string.
   * TODO: saqadri - figure out how to return a string stream for the document instead
   * so large documents aren't all loaded into memory.
   */
  toString(rawDocument: RawDocument): Promise<string>;

  /**
   * Serialize the parsed document to disk, and returns the path to the serialized document
   */
  serialize(rawDocument: RawDocument): Promise<string>;
}

export abstract class BaseDocumentParser implements DocumentParser {
  attributes = {};
  metadata = {};
  callbackManager?: CallbackManager;

  constructor(
    attributes?: JSONObject,
    metadata?: JSONObject,
    callbackManager?: CallbackManager
  ) {
    this.attributes = attributes ?? this.attributes;
    this.metadata = metadata ?? this.metadata;
    this.callbackManager = callbackManager;
  }

  abstract parse(rawDocument: RawDocument): Promise<IngestedDocument>;

  abstract parseNext(
    rawDocument: RawDocument,
    previousFragment?: DocumentFragment,
    take?: number
  ): Promise<DocumentFragment>;

  async toString(rawDocument: RawDocument): Promise<string> {
    return await rawDocument.getContent();
  }

  async serialize(_rawDocument: RawDocument): Promise<string> {
    // TODO: Is this even needed?
    throw new Error("Method not implemented.");
  }
}
