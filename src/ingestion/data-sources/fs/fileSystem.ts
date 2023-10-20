import { JSONObject } from "../../../common/jsonTypes";
import { RawDocument } from "../../../document/document";
import { DataSource } from "../dataSource";
import { LangChainFileLoader } from "./langchainFileLoader";
import { TxtFileLoader } from "./txtFileLoader";
import fs from "fs/promises";
import { extname, resolve } from "node:path";
import * as mime from "mime-types";
import { v4 as uuid } from "uuid";
import { DocxFileLoader } from "./docxFileLoader";
import { PDFFileLoader } from "./pdfFileLoader";
import { CSVFileLoader } from "./csvFileLoader";
import { Md5 } from "ts-md5";

// the import from src/utils/callbacks.ts
import {
  CallbackManager,
  ILoadDocumentsSuccessEvent,
} from "../../../utils/callbacks";

type FileLoaderMap = {
  [extension: string]: (path: string) => LangChainFileLoader;
};

const DEFAULT_FILE_LOADERS: FileLoaderMap = {
  ".csv": (path: string) => new CSVFileLoader(path),
  ".docx": (path: string) => new DocxFileLoader(path),
  ".pdf": (path: string) => new PDFFileLoader(path),
  ".txt": (path: string) => new TxtFileLoader(path),
};

/**
 * A data source that loads documents from a file or directory (recursive) in
 * the local file system.
 */
export class FileSystem implements DataSource {
  name: string = "FileSystem";
  path: string;
  collectionId: string | undefined;
  callbackManager?: CallbackManager;
  fileLoaders: FileLoaderMap = {};

  constructor(
    path: string,
    collectionId?: string,
    fileLoaders?: FileLoaderMap,
    callbackManager?: CallbackManager
  ) {
    this.path = path;
    this.collectionId = collectionId;
    this.fileLoaders = fileLoaders ?? DEFAULT_FILE_LOADERS;
    this.callbackManager = callbackManager ?? undefined;
  }

  private async getStats(): Promise<{
    isDirectory: () => boolean;
    isFile: () => boolean;
  }> {
    const stats = await fs.stat(this.path);
    return {
      isDirectory: () => stats.isDirectory(),
      isFile: () => stats.isFile(),
    };
  }

  private async loadFile(
    filePath: string,
    collectionId?: string
  ): Promise<RawDocument> {
    const extension = extname(filePath);
    const fileLoader = this.fileLoaders[extension];
    if (!fileLoader) {
      throw new Error(`No file loader found for extension ${extension}`);
    }

    const mimeType = mime.lookup(filePath);

    const hash = new Md5();
    const chunks = await fileLoader(filePath).loadChunkedContent();
    for (const chunk of chunks) {
      hash.appendStr(chunk.content + "\n");
    }
    hash.end();

    return {
      uri: filePath,
      dataSource: this,
      name: filePath,
      mimeType: mimeType || "unknown",
      documentId: uuid(),
      collectionId,
      getChunkedContent: async () =>
        await fileLoader(filePath).loadChunkedContent(),
      getContent: async () =>
        await fileLoader(filePath)
          .loadChunkedContent()
          .then((content) => content.join("\n")),
      hash: hash.toString(),
      metadata: {},
      attributes: {},
    };
  }

  async testConnection(): Promise<number> {
    try {
      await this.getStats();
      // If stat succeeds, then the path exists.
      return 200;
    } catch (err) {
      return 404;
    }
  }

  async loadDocuments(
    _filters?: JSONObject,
    _limit?: number | undefined
  ): Promise<RawDocument[]> {
    const { isDirectory, isFile } = await this.getStats();
    const rawDocuments: RawDocument[] = [];

    // TODO: Figure out proper handling of filters and limit (likely BFS)

    if (isDirectory()) {
      const files = await fs.readdir(this.path, { withFileTypes: true });
      const collectionId = this.collectionId ?? uuid();

      const loadDirDocuments = files.map(async (file) => {
        if (file.isDirectory()) {
          const subDir = new FileSystem(
            resolve(this.path, file.name),
            collectionId
          );
          return await subDir.loadDocuments();
        } else if (file.isFile()) {
          return [
            await this.loadFile(resolve(this.path, file.name), collectionId),
          ];
        }
        // If nested path is not a file or directory, just skip for now
      });

      rawDocuments.push(
        ...(await Promise.all(loadDirDocuments))
          .flat()
          .filter((doc): doc is RawDocument => doc != null)
      );
    } else if (isFile()) {
      rawDocuments.push(await this.loadFile(this.path));
    } else {
      throw new Error(`${this.path} is neither a file nor a directory.`);
    }

    if (this.callbackManager !== undefined) {
      const event: ILoadDocumentsSuccessEvent = {
        name: "onLoadDocumentsSuccess",
        data: rawDocuments,
      };
      await this.callbackManager.runCallbacks(event);
    }
    return rawDocuments;
  }
}
