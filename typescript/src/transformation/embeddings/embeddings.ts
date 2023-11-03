import { Attributable } from "../../common/base";
import { JSONObject } from "../../common/jsonTypes";
import { Document, DocumentFragment } from "../../document/document";
import { Transformer } from "../transformer";

export interface VectorEmbedding extends Attributable {
  // The vector representation of the embedding text.
  vector: number[];

  // The text embedded via the vector.
  text: string;
  /**
   * Number of dimensions in the vector, and min/max values for each dimension.
   * This can be used to normalize the vector or checking for out-of-bounds values.
   */
  extras?: {
    dimensions: number;
    min: number;
    max: number;
  };

  metadata: JSONObject & {
    documentId?: string;
    fragmentId?: string;
    retrievalScore?: number;
  };
}

/**
 * Transformer for creating VectorEmbeddings from Documents. Primarily used for
 * indexing and querying vectors via VectorDBs.
 */
export interface EmbeddingsTransformer extends Transformer<VectorEmbedding[]> {
  // The dimensions for the embedding vectors created by this transformer.
  dimensions: number;

  /**
   * Embed the given text and metadata as a vector.
   * @param text The text to embed.
   * @param metadata The metadata to embed.
   */
  embed(text: string, metadata?: JSONObject): Promise<VectorEmbedding>;
}

export abstract class DocumentEmbeddingsTransformer
  implements EmbeddingsTransformer
{
  dimensions: number;

  constructor(dimensions: number) {
    this.dimensions = dimensions;
  }

  /**
   * Embed the given text and metadata as a VectorEmbedding.
   * @param text The text to embed.
   * @param metadata The metadata to embed.
   */
  abstract embed(text: string, metadata?: JSONObject): Promise<VectorEmbedding>;

  /**
   * Embed the given DocumentFragment as a VectorEmbedding.
   * @param text The DocumentFragment to embed.
   */
  protected async embedFragment(
    fragment: DocumentFragment
  ): Promise<VectorEmbedding> {
    const text = await fragment.getContent();
    const metadata = {
      ...fragment.metadata,
      documentId: fragment.documentId,
      fragmentId: fragment.fragmentId,
    };
    return await this.embed(text, metadata);
  }

  /**
   * Embed all fragments for a single document as VectorEmbeddings.
   * This is a convenience method which can be overridden to use the underlying embedding
   * API's batch embedding creation implementation.
   * @param document The document to embed.
   */
  protected async embedDocument(
    document: Document
  ): Promise<VectorEmbedding[]> {
    return await Promise.all(
      document.fragments.map(async (fragment) => this.embedFragment(fragment))
    );
  }

  /**
   * Embed multiple documents as VectorEmbeddings. This is a convenience
   * method which can be overridden to use the underlying embedding API's recommended
   * batch embedding creation implementation.
   * @param documents The documents to embed.
   */
  async transformDocuments(documents: Document[]): Promise<VectorEmbedding[]> {
    const embeddings = await Promise.all(
      documents.map((document) => this.embedDocument(document))
    );
    return embeddings.flat();
  }
}
