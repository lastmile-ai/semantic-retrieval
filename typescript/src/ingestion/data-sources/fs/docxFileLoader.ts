import { DocxLoader } from "langchain/document_loaders/fs/docx";
import {
  LangChainFileLoader,
  LangChainFileLoaderOptions,
} from "./langchainFileLoader";

export class DocxFileLoader extends LangChainFileLoader {
  constructor(path: string, options?: LangChainFileLoaderOptions) {
    super(path, new DocxLoader(path), options);
  }
}
