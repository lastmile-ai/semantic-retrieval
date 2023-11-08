import { TextLoader } from "langchain/document_loaders/fs/text";
import {
  LangChainFileLoader,
  LangChainFileLoaderOptions,
} from "./langchainFileLoader";

export class TxtFileLoader extends LangChainFileLoader {
  constructor(path: string, options?: LangChainFileLoaderOptions) {
    super(path, new TextLoader(path), options);
  }
}
