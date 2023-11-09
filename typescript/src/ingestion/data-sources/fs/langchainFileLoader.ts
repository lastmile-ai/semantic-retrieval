import { BaseDocumentLoader } from "langchain/dist/document_loaders/base";
import { BaseFileLoader } from "./fileLoader";
import { RawDocumentChunk } from "../../../document/document";
import {
  CallbackManager,
  LoadChunkedContentEvent,
} from "../../../utils/callbacks";

export type LangChainFileLoaderOptions = {
  callbackManager?: CallbackManager;
};

/**
 * Abstract class for loading RawDocuments from a file using
 * LangChain file-based DocumentLoaders.
 */
export abstract class LangChainFileLoader extends BaseFileLoader {
  loader: BaseDocumentLoader;
  callbackManager?: CallbackManager;

  constructor(
    path: string,
    loader: BaseDocumentLoader,
    options?: LangChainFileLoaderOptions
  ) {
    super(path);
    this.loader = loader;
    this.callbackManager = options?.callbackManager;
  }

  // LangChain loaders return Document object(s) with pageContent and metadata.
  // A single file may contain multiple documents, depending on how the file
  // is chunked (e.g. by pages)
  async loadChunkedContent(): Promise<RawDocumentChunk[]> {
    const documents = await this.loader.load();
    const chunkedContent = documents.map((document) => ({
      content: document.pageContent,
      metadata: document.metadata,
    }));

    const event: LoadChunkedContentEvent = {
      name: "onLoadChunkedContent",
      chunkedContent,
      path: this.path,
      loader: this.loader,
    };

    await this.callbackManager?.runCallbacks(event);

    return chunkedContent;
  }
}
