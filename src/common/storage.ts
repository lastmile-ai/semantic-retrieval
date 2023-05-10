import {
  createReadStream,
  createWriteStream,
  mkdtempSync,
  promises as fs,
} from "fs";
import { Readable } from "stream";
import sanitize from "sanitize-filename";
import * as path from "path";
import { v4 as uuid } from "uuid";

/**
 * A storage interface for storing and retrieving data during ingestion and processing.
 * For example, documents, document fragments, transformed data, can be stored in during a pipeline run.
 */
export interface BlobStorage {
  /**
   * Saves the blob to the storage.
   * @param blob The blob to save.
   * @param name The name of the blob.
   * @returns A promise that resolves to the ID of the saved blob.
   */
  write(blob: Blob, name?: string): Promise<BlobIdentifier>;

  /**
   * Writes the blob to the storage from a stream.
   */
  writeStream(stream: Readable, name?: string): Promise<BlobIdentifier>;

  /**
   * Gets the blob from the storage.
   */
  read(blobUri: string): Promise<Blob>;

  /**
   * Streaming data from the blob storage.
   */
  readStream(blobUri: string): Promise<Readable>;

  /**
   * Deletes the blob from the storage.
   */
  delete(blobUri: string): Promise<void>;
}

/**
 * A blob identifier is a unique identifier for a blob in a blob storage.
 */
export interface BlobIdentifier {
  blobUri: string;
  storage: BlobStorage;
}

/**
 * A blob is a piece of data that can be stored in a BlobStorage.
 */
export interface Blob {
  data: Uint8Array;
  mimeType?: string;
  id: BlobIdentifier;
}

/**
 * A blob storage that stores data in the local file system.
 */
export class FileSystemStorage implements BlobStorage {
  private workingDirectory: string;

  constructor(workingDirectory: string) {
    this.workingDirectory = workingDirectory;
  }

  async write(blob: Blob, name?: string): Promise<BlobIdentifier> {
    const fileName = sanitize(name || uuid());
    const filePath = path.join(this.workingDirectory, fileName);
    await fs.writeFile(filePath, blob.data);

    return {
      blobUri: filePath,
      storage: this,
    };
  }

  async writeStream(stream: Readable, name?: string): Promise<BlobIdentifier> {
    const fileName = sanitize(name || uuid());
    const filePath = path.join(this.workingDirectory, fileName);

    return new Promise<BlobIdentifier>((resolve, reject) => {
      const writeStream = createWriteStream(filePath);
      stream.pipe(writeStream);
      writeStream.on("error", reject);
      writeStream.on("finish", () => {
        resolve({
          blobUri: filePath,
          storage: this,
        });
      });
    });
  }

  async read(blobUri: string): Promise<Blob> {
    const data = await fs.readFile(blobUri);
    return {
      data,
      id: {
        blobUri,
        storage: this,
      },
    };
  }

  async readStream(blobUri: string): Promise<Readable> {
    const stream = createReadStream(blobUri);
    return stream;
  }
  async delete(blobUri: string): Promise<void> {
    await fs.unlink(blobUri);
  }
}

/**
 * A blob storage that stores data in a temporary file system.
 * Data gets destroyed when the process exits.
 */
export class TemporaryFileSystemStorage extends FileSystemStorage {
  static instance: TemporaryFileSystemStorage =
    new TemporaryFileSystemStorage();

  constructor() {
    const workingDirectory = mkdtempSync("temporary-file-system-storage");
    super(workingDirectory);
  }
}

/**
 * A blob storage that stores data in an AWS S3 bucket.
 */
export class S3Storage implements BlobStorage {
  constructor() {}
  write(blob: Blob, name?: string): Promise<BlobIdentifier> {
    throw new Error("Method not implemented.");
  }
  writeStream(stream: Readable, name?: string): Promise<BlobIdentifier> {
    throw new Error("Method not implemented.");
  }
  read(blobUri: string): Promise<Blob> {
    throw new Error("Method not implemented.");
  }
  readStream(blobUri: string): Promise<Readable> {
    throw new Error("Method not implemented.");
  }
  delete(blobUri: string): Promise<void> {
    throw new Error("Method not implemented.");
  }
}
