import type { BaseDocumentLoader } from "langchain/document_loaders/base";
import { BaseFileLoader } from "./fileLoader.js";
import { RawDocumentChunks } from "../../../document/document.js";

/**
 * Abstract class for loading RawDocuments from a file using
 * LangChain file-based DocumentLoaders.
 */
export abstract class LangChainFileLoader extends BaseFileLoader {
  loader: BaseDocumentLoader;

  constructor(path: string, loader: BaseDocumentLoader) {
    super(path);
    this.loader = loader;
  }

  // LangChain loaders return Document object(s) with pageContent and metadata.
  // A single file may contain multiple documents, depending on how the file
  // is chunked (e.g. by pages)
  async loadChunkedContent(): Promise<RawDocumentChunks> {
    const documents = await this.loader.load();
    return documents.map((document) => ({
      content: document.pageContent,
      metadata: document.metadata,
    }));
  }
}
