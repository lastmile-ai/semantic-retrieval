import { Document } from "../document/document";

/**
 * A Transformer can transform documents into other types for different
 * use cases. Common transformations include text chunking, summarization
 * and embeddings.
 */
export interface Transformer<T> {
  transformDocuments(documents: Document[]): Promise<T>;
}
