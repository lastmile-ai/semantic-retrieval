import { JSONObject } from "../../../common/jsonTypes.js";
import { RawDocument } from "../../../document/document.js";
import { DataSource } from "../dataSource.js";
import { LangChainFileLoader } from "./langchainFileLoader.js";
import { TxtFileLoader } from "./txtFileLoader.js";
import fs from "fs/promises";
import { extname, resolve } from "node:path";
import * as mime from "mime-types";
import { v4 as uuid } from "uuid";
import { DocxFileLoader } from "./docxFileLoader.js";
import { PDFFileLoader } from "./pdfFileLoader.js";
import { CSVFileLoader } from "./csvFileLoader.js";

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
  fileLoaders: FileLoaderMap = {};

  constructor(
    path: string,
    collectionId?: string,
    fileLoaders?: FileLoaderMap
  ) {
    this.path = path;
    this.collectionId = collectionId;
    this.fileLoaders = fileLoaders ?? DEFAULT_FILE_LOADERS;
  }

  private async getStats(): Promise<{
    isDirectory: () => boolean;
    isFile: () => boolean;
  }> {
    return await fs.stat(this.path);
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

      for (const file of files) {
        if (file.isDirectory()) {
          const subDir = new FileSystem(
            resolve(this.path, file.name),
            collectionId
          );
          rawDocuments.push(...(await subDir.loadDocuments()));
        } else if (file.isFile()) {
          rawDocuments.push(
            await this.loadFile(resolve(this.path, file.name), collectionId)
          );
        }
        // If nested path is not a file or directory, just skip for now
      }
    } else if (isFile()) {
      rawDocuments.push(await this.loadFile(this.path));
    } else {
      throw new Error(`${this.path} is neither a file nor a directory.`);
    }

    return rawDocuments;
  }
}
