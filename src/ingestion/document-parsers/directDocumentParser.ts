import { JSONObject } from "../../common/jsonTypes";
import {
  RawDocument,
  IngestedDocument,
  DocumentFragment,
} from "../../document/document";
import { BaseDocumentParser } from "./documentParser";
import { v4 as uuid } from "uuid";
import { Md5 } from "ts-md5";
import {
  CallbackManager,
  ParseNextErrorEvent,
  ParseSuccessEvent,
} from "../../utils/callbacks";

/**
 * Parse a RawDocument directly into a Document, with each DocumentFragment
 * representing a RawDocumentChunk.
 */
export class DirectDocumentParser extends BaseDocumentParser {
  constructor(
    attributes?: JSONObject,
    metadata?: JSONObject,
    callbackManager?: CallbackManager
  ) {
    super(attributes, metadata, callbackManager);
  }

  async parse(rawDocument: RawDocument): Promise<IngestedDocument> {
    const chunks = await rawDocument.getChunkedContent();
    const documentId = uuid();

    const fragments: DocumentFragment[] = [];

    for (const [idx, chunk] of chunks.entries()) {
      const currentFragment: DocumentFragment = {
        fragmentId: uuid(),
        fragmentType: "text",
        documentId,
        metadata: chunk.metadata,
        attributes: {},
        // TODO: figure out blobId
        hash: Md5.hashStr(chunk.content),
        getContent: async () => chunk.content,
        serialize: async () => JSON.stringify(chunk),
        previousFragment: idx > 0 ? fragments[idx - 1] : undefined,
      };

      if (idx > 0) {
        fragments[idx - 1].nextFragment = currentFragment;
      }

      fragments.push(currentFragment);
    }

    const out = {
      documentId,
      rawDocument,
      collectionId: rawDocument.collectionId,
      fragments,
      serialize: () => this.serialize(rawDocument),
      attributes: this.attributes,
      metadata: this.metadata,
    };
    const event: ParseSuccessEvent = {
      name: "onParseSuccess",
      ingestedDocument: out,
    };
    this.callbackManager?.runCallbacks(event);
    return out;
  }

  // TODO: This is mainly useful for parsing the raw document as a stream/buffer
  // instead of chunking the full document at once when loading.
  // Revisit when we add support for that
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
}
