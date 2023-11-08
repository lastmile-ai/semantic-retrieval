import { RawDocumentChunk } from "../../../document/document";
import { CallbackManager, Traceable } from "../../../utils/callbacks";

/**
 * Abstract class for loading chunked content from a file.
 */
export abstract class BaseFileLoader implements Traceable {
  path: string;
  callbackManager?: CallbackManager;

  constructor(path: string, callbackManager?: CallbackManager) {
    this.path = path;
    this.callbackManager = callbackManager;
  }

  abstract loadChunkedContent(): Promise<RawDocumentChunk[]>;
}
