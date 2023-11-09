import { BaseDocumentTransformer } from "../documentTransformer";
import { Document, DocumentFragment } from "../../../document/document";
import { DocumentMetadataDB } from "../../../document/metadata/documentMetadataDB";
import { v4 as uuid } from "uuid";
import { Md5 } from "ts-md5";
import { Traceable, TransformDocumentEvent } from "../../../utils/callbacks";

export interface TextChunkConfig {
  // The size limit of each chunk, measured by sizeFn. Each chunk (including overlap)
  // should not exceed this limit, but may be smaller
  chunkSizeLimit?: number;
  // When constructing a new chunk from sub-chunks and the first sub-chunk is small enough,
  // the new chunk will have the latest sub-chunks from the previous chunk prepended to it until
  // chunkOverlap is reached. This allows for more context to be included in each chunk.
  chunkOverlap?: number;
  // Determines how size is measured for the text. Defaults to text length.
  sizeFn?: (text: string) => Promise<number>;
}

export type TextChunkTransformerParams = TextChunkConfig &
  Traceable & {
    metadataDB?: DocumentMetadataDB;
  };

/**
 * A DocumentTransformer that splits a Document's fragments into chunks of
 * text of a maximum size (chunkSize). The chunking operation consists of a
 * series of splits (chunk larger text into smaller sub-chunks) and merges (merge
 * smaller sub-chunks into one larger chunk).
 */
export abstract class TextChunkTransformer
  extends BaseDocumentTransformer
  implements TextChunkConfig
{
  chunkSizeLimit = 500;
  chunkOverlap = 100;
  sizeFn = async (text: string) => text.length;

  constructor(params?: TextChunkTransformerParams) {
    super(params?.metadataDB, params?.callbackManager);
    this.chunkSizeLimit = params?.chunkSizeLimit ?? this.chunkSizeLimit;
    this.chunkOverlap = params?.chunkOverlap ?? this.chunkOverlap;
    this.sizeFn = params?.sizeFn ?? this.sizeFn;

    if (this.chunkOverlap >= this.chunkSizeLimit) {
      throw new Error("chunkOverlap must be less than chunkSizeLimit");
    }
  }

  abstract chunkText(text: string): Promise<string[]>;

  protected subChunkOnSeparator(text: string, separator: string): string[] {
    const subChunks = text.split(separator);
    return subChunks.filter((sc) => sc !== "");
  }

  async transformDocument(document: Document): Promise<Document> {
    const originalFragmentsData = await Promise.all(
      document.fragments.map(async (fragment) => {
        const content = await fragment.getContent();
        return {
          attributes: fragment.attributes,
          content,
          fragmentType: fragment.fragmentType,
          metadata: fragment.metadata,
        };
      })
    );

    const transformedFragments: DocumentFragment[] = [];
    let fragmentCount = 0;
    const documentId = uuid();

    for (let i = 0; i < originalFragmentsData.length; i++) {
      const originalFragment = originalFragmentsData[i];

      for (const chunk of await this.chunkText(originalFragment.content)) {
        const currentFragment: DocumentFragment = {
          fragmentId: uuid(),
          fragmentType: "text",
          documentId,
          // TODO: Figure out if/how we should add additional metadata from the chunks (e.g. line count)
          metadata: originalFragment.metadata,
          attributes: {},
          // TODO: figure out blobId
          hash: Md5.hashStr(chunk),
          getContent: async () => chunk,
          serialize: async () => chunk,
          previousFragment:
            fragmentCount > 0
              ? transformedFragments[fragmentCount - 1]
              : undefined,
        };

        if (fragmentCount > 0) {
          transformedFragments[fragmentCount - 1].nextFragment =
            currentFragment;
        }

        fragmentCount++;
        transformedFragments.push(currentFragment);
      }
    }

    const transformedDocument = {
      ...document,
      documentId,
      fragments: transformedFragments,
    };

    const event: TransformDocumentEvent = {
      name: "onTransformDocument",
      originalDocument: document,
      transformedDocument,
    };
    await this.callbackManager?.runCallbacks(event);

    // TODO: Think through metadata handling, since setting new doc metadata on each transformation
    // can cause proliferation of DB entries. On the other hand, we probably don't want to mutate
    // the document in place since we may want to perform different operations on the same document.
    // With optional metadata DB, we can just pass it through in the 'final' transformations / those
    // we want to persist document metadata for. But, access policies are currently constructed from the raw document
    // during parsing, so we'd need to support the same policy factory stuff in each transformer...
    return transformedDocument;
  }

  protected joinSubChunks(
    subChunks: string[],
    separator: string
  ): string | null {
    const chunk = subChunks.join(separator).trim();
    return chunk === "" ? null : chunk;
  }

  protected async mergeSubChunks(
    subChunks: string[],
    separator: string
  ): Promise<string[]> {
    const chunks: string[] = [];
    let prevSubChunks: string[] = [];
    let currentSubChunks: string[] = [];
    let currentChunkSize = 0;

    const chunkSeparatorSize = await this.sizeFn(separator);

    for (const sc of subChunks) {
      const scSize = await this.sizeFn(sc);

      if (scSize > this.chunkSizeLimit) {
        console.warn(
          `SubChunk size ${scSize} exceeds chunkSizeLimit of ${this.chunkSizeLimit}`
        );
      }

      if (
        currentChunkSize + chunkSeparatorSize + scSize >
        this.chunkSizeLimit
      ) {
        // Current subChunk would cause the chunk (with separators) to exceed the size limit, so
        // create the current chunk before starting a new one with the current subChunk
        const chunk = this.joinSubChunks(currentSubChunks, separator);
        if (chunk != null) {
          chunks.push(chunk);
        }
        prevSubChunks = currentSubChunks;
        currentChunkSize = 0;
        currentSubChunks = [];
      }

      // If this is the first subChunk after a completed chunk, handle chunk overlap from prev chunk
      const numTotalPrevSubChunks = prevSubChunks.length;
      if (currentSubChunks.length === 0 && numTotalPrevSubChunks > 0) {
        let prevSubChunksOverlapSize = 0;
        let numPrevSubChunksOverlap = 0;

        // We take as many previous subchunks that will fit into the overlap along with at least one current
        // chunk, as long as they all fit within the chunk size limit
        while (numPrevSubChunksOverlap < numTotalPrevSubChunks) {
          const nextPrevSubChunkSize = await this.sizeFn(
            prevSubChunks[numTotalPrevSubChunks - numPrevSubChunksOverlap - 1]
          );

          if (
            prevSubChunksOverlapSize +
              chunkSeparatorSize +
              nextPrevSubChunkSize >
              this.chunkOverlap ||
            prevSubChunksOverlapSize +
              chunkSeparatorSize +
              nextPrevSubChunkSize +
              scSize >
              this.chunkSizeLimit
          ) {
            // Adding this prev subchunk would exceed the overlap or chunk size limit, so stop
            break;
          }

          prevSubChunksOverlapSize += nextPrevSubChunkSize;
          if (numPrevSubChunksOverlap > 0) {
            // Separator only added between subchunks, so skip for first subchunk
            prevSubChunksOverlapSize += chunkSeparatorSize;
          }
          numPrevSubChunksOverlap++;
        }

        while (numPrevSubChunksOverlap > 0) {
          currentSubChunks.push(
            prevSubChunks[numTotalPrevSubChunks - numPrevSubChunksOverlap]
          );
          numPrevSubChunksOverlap--;
        }

        currentChunkSize = prevSubChunksOverlapSize;
      }

      currentSubChunks.push(sc);
      if (currentSubChunks.length > 1) {
        // Separator only added between subchunks, so skip for first subchunk
        currentChunkSize += chunkSeparatorSize;
      }
      currentChunkSize += scSize;
    }

    // Close up the final chunk if needed
    if (currentSubChunks.length > 0) {
      const chunk = this.joinSubChunks(currentSubChunks, separator);
      if (chunk != null) {
        chunks.push(chunk);
      }
    }

    return chunks;
  }
}
