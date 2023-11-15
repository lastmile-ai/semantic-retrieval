import { CSVLoader } from "langchain/document_loaders/fs/csv";
import {
  LangChainFileLoader,
  LangChainFileLoaderOptions,
} from "./langchainFileLoader";

type LangChainCSVLoaderOptions = LangChainFileLoaderOptions & {
  column?: string;
  separator?: string;
};

export class CSVFileLoader extends LangChainFileLoader {
  constructor(path: string, options?: LangChainCSVLoaderOptions) {
    super(path, new CSVLoader(path, options), options);
  }
}
