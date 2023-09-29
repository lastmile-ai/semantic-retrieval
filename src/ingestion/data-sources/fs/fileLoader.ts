import { RawDocumentChunk } from "../../../document/document";

/**
 * Abstract class for loading chunked content from a file.
 */
export abstract class BaseFileLoader {
    path: string;
  
    constructor(path: string) {
      this.path = path;
    }
  
    abstract loadChunkedContent(): Promise<RawDocumentChunk[]>;
  }