import { BaseDocumentLoader } from "langchain/dist/document_loaders/base";
import { BaseFileLoader } from "./fileLoader";
import { RawDocumentChunk } from "../../../document/document";

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
  async loadChunkedContent(): Promise<RawDocumentChunk[]> {
    const documents = await this.loader.load();
    return documents.map((document) => ({
      content: document.pageContent,
      metadata: document.metadata,
    }));
  }
}
