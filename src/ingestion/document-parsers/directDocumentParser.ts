import { JSONObject } from "../../common/jsonTypes";
import {
  RawDocument,
  Document,
  DocumentFragment,
} from "../../document/document";
import { BaseDocumentParser } from "./documentParser";
import { v4 as uuid } from "uuid";
import { Md5 } from "ts-md5";

/**
 * Parse a RawDocument directly into a Document, with each DocumentFragment
 * representing a RawDocumentChunk.
 */
export class DirectDocumentParser extends BaseDocumentParser {
  constructor(attributes?: JSONObject, metadata?: JSONObject) {
    super(attributes, metadata);
  }

  async parse(rawDocument: RawDocument): Promise<Document> {
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

    return {
      documentId,
      rawDocument,
      collectionId: rawDocument.collectionId,
      fragments,
      serialize: () => this.serialize(rawDocument),
      attributes: this.attributes,
      metadata: this.metadata,
    };
  }

  // TODO: This is mainly useful for parsing the raw document as a stream/buffer
  // instead of chunking the full document at once when loading.
  // Revisit when we add support for that
  parseNext(
    _rawDocument: RawDocument,
    _previousFragment?: DocumentFragment,
    _take?: number
  ): Promise<DocumentFragment> {
    throw new Error("Not implemented");
  }
}