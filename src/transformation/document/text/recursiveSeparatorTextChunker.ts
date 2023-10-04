import {
  TextChunkConfig,
  TextChunkTransformer,
  TextChunkTransformerParams,
} from "./textChunkTransformer";

export interface RecursiveSeparatorTextChunkConfig extends TextChunkConfig {
  separators: string[];
}

export type RecursiveSeparatorTextChunkerParams = TextChunkTransformerParams & {
  config?: RecursiveSeparatorTextChunkConfig;
};

/**
 * Chunk text by prioritized separators. For a given chunkSize, the chunker will attempt
 * to split-and-merge the entire text into chunks of at most chunkSize using the first separator.
 * If any chunk from the split using the highest priority separator is larger than chunkSize,
 * the next highest priority separator is used to split-and-merge the remaining text (and so on),
 * until the chunk size is met for all sub-chunks (or the last separator is attempted).
 * Caution: If no separator provided can split the text into chunks of the desired size, the resulting
 * chunks may exceed chunkSize.
 *
 *  The default separators, in order, are:
 * 1. "\n\n" - paragraph
 * 2. "\n" - line
 * 3. " " - word
 * 4. "" - character
 */
export class RecursiveSeparatorTextChunker extends TextChunkTransformer {
  separators: string[] = ["\n\n", "\n", " ", ""];

  constructor(params?: RecursiveSeparatorTextChunkerParams) {
    super(params);
    this.separators = params?.config?.separators ?? this.separators;
  }

  private async chunkTextRecursive(
    text: string,
    separators: string[]
  ): Promise<string[]> {
    let separator = separators[separators.length - 1];
    let nextSeparators;

    // Find the highest priority separator that exists in the text and
    // get the subset of next priority separators to use on potential
    // sub-chunks of the text
    for (let i = 0; i < separators.length; i++) {
      if (text.includes(separators[i])) {
        separator = separators[i];
        if (separator !== "") {
          // Can't reduce single character subchunks any further
          nextSeparators = separators.slice(i + 1);
        }
        break;
      }
    }

    const subChunks = this.subChunkOnSeparator(text, separator);
    const chunksToMerge: string[] = [];

    for (const subChunk of subChunks) {
      if ((await this.sizeFn(subChunk)) < this.chunkSizeLimit) {
        chunksToMerge.push(subChunk);
      } else {
        if (!nextSeparators) {
          chunksToMerge.push(subChunk);
        } else {
          const recursiveSubChunks = await this.chunkTextRecursive(
            subChunk,
            nextSeparators
          );
          chunksToMerge.push(...recursiveSubChunks);
        }
      }
    }

    return await this.mergeSubChunks(chunksToMerge);
  }

  chunkText(text: string): Promise<string[]> {
    return this.chunkTextRecursive(text, this.separators);
  }
}
