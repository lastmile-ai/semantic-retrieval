import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import {
  LangChainFileLoader,
  LangChainFileLoaderOptions,
} from "./langchainFileLoader";

export class PDFFileLoader extends LangChainFileLoader {
  constructor(path: string, options?: LangChainFileLoaderOptions) {
    super(path, new PDFLoader(path), options);
  }
}
