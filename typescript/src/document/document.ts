import { Attributable } from "../common/base";
import { JSONObject } from "../common/jsonTypes";
import { BlobIdentifier } from "../common/storage";
import { DataSource } from "../ingestion/data-sources/dataSource";

export type RawDocumentChunk = { content: string; metadata: JSONObject };

/**
 * The original document, as it was ingested from the data source.
 */
export interface RawDocument extends Attributable {
  uri: string;
  dataSource: DataSource;
  name: string;
  // Mime type of the document or "unknown" if not known.
  mimeType: string;
  // The hash of the document content.
  hash?: string;
  // Storage path to the raw document file, if it has been saved.
  blobId?: BlobIdentifier;

  // Unique identifier for the source document.
  documentId: string;

  // Unique identifier for the collection that this document belongs to,
  // if applicable.
  collectionId?: string;

  /**
   * Fetch the document text from the data source.
   * TODO: saqadri - instead of fetching the entire document text (which could be large), we should
   * provide a way to fetch the text in chunks or stream the text.
   * Perhaps readable stream with async generators: https://nodejs.org/api/stream.html#creating-readable-streams-with-async-generators
   */
  getContent(): Promise<string>;

  /**
   * Fetch the document text and metadata in reasonable chunks (e.g. pages) from the data source.
   */
  getChunkedContent(): Promise<RawDocumentChunk[]>;
}

export type DocumentFragmentType =
  | "text"
  | "image"
  | "table"
  | "list"
  | "paragraph"
  | "heading"
  | "code"
  | "quote";

/**
 *
 */
export interface DocumentFragment extends Attributable {
  fragmentId: string;
  // The hash of the fragment content.
  hash?: string;
  // Storage path to the fragment content
  blobId?: BlobIdentifier;

  fragmentType: DocumentFragmentType;

  // The ID for the document that this fragment belongs to.
  documentId: string;
  // The previous fragment in the document.
  previousFragment?: DocumentFragment;
  // The next fragment in the document.
  nextFragment?: DocumentFragment;
  // Sub-fragments within this fragment.
  children?: DocumentFragment[];

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
export interface Document extends Attributable {
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

/**
 * A @see RawDocument after it has been parsed into a graph of @see DocumentFragments during
 * ingestion; RawDocument is accessible from the Document at ingestion time.
 */
export interface IngestedDocument extends Document {
  rawDocument: RawDocument;
}
