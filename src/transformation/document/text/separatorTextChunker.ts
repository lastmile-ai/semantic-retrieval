import { TextChunkConfig, TextChunkTransformer, TextChunkTransformerParams } from "./textChunkTransformer";

export interface SeparatorTextChunkConfig extends TextChunkConfig {
    separator: string
  }

  export type SeparatorTextChunkerParams = TextChunkTransformerParams &{
    config?: SeparatorTextChunkConfig;
  }

/**
 * Chunk text by a specified separator. The default separator is "\n", which
 * will chunk simple text into words.
 */
export class SeparatorTextChunker extends TextChunkTransformer {
    separator: string = " "; // e.g. words
    
    constructor(params?: SeparatorTextChunkerParams) {
        super(params);
        this.separator = params?.config?.separator ?? this.separator;
    }
    
    chunkText(text: string): Promise<string[]> {
        const subChunks = this.subChunkOnSeparator(text, this.separator);
        return this.mergeSubChunks(subChunks);
    }

} 