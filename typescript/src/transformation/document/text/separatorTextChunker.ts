import { ChunkTextEvent } from "../../../utils/callbacks";
import {
  TextChunkConfig,
  TextChunkTransformer,
  TextChunkTransformerParams,
} from "./textChunkTransformer";

export interface SeparatorTextChunkConfig extends TextChunkConfig {
  separator?: string;
  stripNewlines?: boolean;
}

export type SeparatorTextChunkerParams = TextChunkTransformerParams &
  SeparatorTextChunkConfig;

/**
 * Chunk text by a specified separator. The default separator is "\n", which
 * will chunk simple text into words.
 */
export class SeparatorTextChunker extends TextChunkTransformer {
  separator: string = " "; // e.g. words
  stripNewlines = true;

  constructor(params?: SeparatorTextChunkerParams) {
    super(params);
    this.separator = params?.separator ?? this.separator;
    this.stripNewlines =
      params?.stripNewlines ?? (this.stripNewlines && this.separator !== "\n");
  }

  async chunkText(text: string): Promise<string[]> {
    let textToChunk = text;
    if (this.stripNewlines) {
      textToChunk = text.replace(/\n/g, " ");
    }
    const subChunks = this.subChunkOnSeparator(textToChunk, this.separator);
    const mergedChunks = await this.mergeSubChunks(subChunks, this.separator);

    const event: ChunkTextEvent = {
      name: "onChunkText",
      chunks: mergedChunks,
    };
    await this.callbackManager?.runCallbacks(event);

    return mergedChunks;
  }
}
