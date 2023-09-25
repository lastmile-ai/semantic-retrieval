import { JSONObject } from "../../../common/jsonTypes";
import { RawDocument } from "../../../document/document";
import { DataSource } from "../dataSource";
import { LangChainFileLoader } from "./langchainFileLoader";
import { TxtFileLoader } from "./txtFileLoader";
import fs from "fs/promises";

type FileLoaderMap = { [extension: string]: (path: string) => LangChainFileLoader };

export class FileSystem implements DataSource {
    name: string = "FileSystem";
    path: string;
    fileLoaders: FileLoaderMap = {};
  
    constructor(path: string) {
      this.path = path;
      this.fileLoaders = {
        "txt": (path: string) => new TxtFileLoader(path),
      }
    }
  
    async testConnection(): Promise<number> {
      try {
        await fs.stat(this.path);
        // If stat succeeds, then the path exists.
        return 200;
      } catch (err) {
        return 404;
      }
    }

    loadDocuments(
      _filters?: JSONObject,
      _limit?: number | undefined,
    ): Promise<RawDocument[]> {
      throw new Error("Method not implemented.");
    }
  }