import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { LangChainFileLoader } from "./langchainFileLoader.js";

export class PDFFileLoader extends LangChainFileLoader {
  constructor(path: string) {
    super(path, new PDFLoader(path));
  }
}
