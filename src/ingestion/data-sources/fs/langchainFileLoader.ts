import { BaseDocumentLoader } from "langchain/dist/document_loaders/base";
import { BaseFileLoader } from "./fileLoader";

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

  // LangChain loaders return Document object(s) with pageContent. A single file
  // may contain multiple documents, so we need to concatenate the pageContent
  async loadContent(): Promise<string> {
    const documents = await this.loader.load();
    return documents.map((document) => document.pageContent).join("\n");
  }
}
