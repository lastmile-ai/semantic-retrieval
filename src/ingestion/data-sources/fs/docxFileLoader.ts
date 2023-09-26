import { DocxLoader } from "langchain/document_loaders/fs/docx";
import { LangChainFileLoader } from "./langchainFileLoader.js";

export class DocxFileLoader extends LangChainFileLoader {
  constructor(path: string) {
    super(path, new DocxLoader(path));
  }
}
