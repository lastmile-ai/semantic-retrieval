import { BlobIdentifier } from "../common/storage";
import { DataSource } from "../ingestion/data-sources/dataSource";

/**
 * The original document, as it was ingested from the data source.
 */
export interface RawDocument {
  uri: string;
  dataSource: DataSource;
  name: string;
  mimeType: string;
  // The hash of the document content.
  hash?: string;
  // Storage path to the raw document file, if it has been saved.
  blobId?: BlobIdentifier;

  // Any JSON-serializable metadata associated with the document.
  metadata: { [key: string]: string };
  // A general property bag associated with this object.
  attributes: { [key: string]: string };

  // Some identifiers for the document.
  documentId: string;
  collectionId?: string;

  /**
   * Fetches the document text from the data source.
   */
  getDocument(): Promise<string>;

  /**
   * Get the document text.
   * TODO: saqadri - instead of fetching the entire document text (which could be large), we should
   * provide a way to fetch the text in chunks or stream the text.
   * Perhaps readable stream with async generators: https://nodejs.org/api/stream.html#creating-readable-streams-with-async-generators
   */
  getContent(): Promise<string>;
}

/**
 *
 */
export interface DocumentFragment {
  fragmentId: string;
  // The hash of the fragment content.
  hash?: string;
  // Storage path to the fragment content
  blobId?: BlobIdentifier;

  fragmentType:
    | "text"
    | "image"
    | "table"
    | "list"
    | "paragraph"
    | "heading"
    | "code"
    | "quote";

  // The document that this fragment belongs to.
  document: Document;
  // The previous fragment in the document.
  previousFragment?: DocumentFragment;
  // The next fragment in the document.
  nextFragment?: DocumentFragment;
  // Sub-fragments within this fragment.
  children?: DocumentFragment[];

  // Any JSON-serializable metadata associated with the document fragment.
  metadata?: { [key: string]: string };
  // A general property bag associated with this object.
  attributes?: { [key: string]: string };

  /**
   * Gets the content of the fragment. @see blobId
   */
  getContent(): Promise<string>;

  /**
   * Serializes the fragment and its relationship metadata as a JSON string.
   */
  serialize(): Promise<string>;
}

/**
 * A @see RawDocument after it has been parsed into a graph of @see DocumentFragments.
 */
export interface Document {
  rawDocument: RawDocument;

  // Any JSON-serializable metadata associated with the document.
  metadata: { [key: string]: string };
  // A general property bag associated with this object.
  attributes: { [key: string]: string };

  // Some identifiers for the document (these could be different from the raw document identifiers)
  documentId: string;
  collectionId?: string;

  // The root fragments of the document.
  fragments: DocumentFragment[];

  /**
   * Serializes the document to disk, and returns the path to the serialized document.
   */
  serialize(): Promise<string>;
}
