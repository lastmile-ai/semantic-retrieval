import { TextLoader } from "langchain/document_loaders/fs/text";
import { LangChainFileLoader } from "./langchainFileLoader";

export class TxtFileLoader extends LangChainFileLoader {
  constructor(path: string) {
    super(path, new TextLoader(path));
  }
}
