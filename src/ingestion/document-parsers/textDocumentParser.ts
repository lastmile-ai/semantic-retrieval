import { JSONObject } from "../../common/jsonTypes";
import {
  RawDocument,
  DocumentFragment,
  IngestedDocument,
} from "../../document/document";
import {
  CallbackManager,
  ParseErrorEvent,
  ParseNextErrorEvent,
} from "../../utils/callbacks";
import { BaseDocumentParser } from "./documentParser";

/**
 * A basic DocumentParser implementation for text/plain documents.
 */
export class TextDocumentParser extends BaseDocumentParser {
  attributes = {};
  metadata = {};
  callbackManager?: CallbackManager;

  constructor(
    attributes?: JSONObject,
    metadata?: JSONObject,
    callbackManager?: CallbackManager
  ) {
    super(attributes, metadata, callbackManager);
  }

  // TODO: Actually implement this when we have txt files loaded from non-langchain-directory sources
  async parse(_rawDocument: RawDocument): Promise<IngestedDocument> {
    const err = new Error("Method not implemented.");
    const event: ParseErrorEvent = {
      name: "onParseError",
      error: err,
    };
    this.callbackManager?.runCallbacks(event);
    throw err;
  }

  async parseNext(
    _rawDocument: RawDocument,
    _previousFragment?: DocumentFragment,
    _take?: number
  ): Promise<DocumentFragment> {
    const err = new Error("Method not implemented.");
    const event: ParseNextErrorEvent = {
      name: "onParseNextError",
      error: err,
    };
    await this.callbackManager?.runCallbacks(event);
    throw err;
  }

  toString(_rawDocument: RawDocument): Promise<string> {
    throw new Error("Method not implemented.");
  }
}
