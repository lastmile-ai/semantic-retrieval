import { Attributable } from "../../common/base";
import { Document, DocumentFragment } from "../../document/document";
import { Transformer } from "../transformer";

export interface VectorEmbedding extends Attributable {
  vector: number[];
  /**
   * Number of dimensions in the vector, and min/max values for each dimension.
   * This can be used to normalize the vector or checking for out-of-bounds values.
   */
  extras?: {
    dimensions: number;
    min: number;
    max: number;
  };
}

/**
 * Transformer for creating VectorEmbeddings from Documents. Primarily used for
 * indexing and querying vectors via VectorDBs.
 */
export interface EmbeddingsTransformer extends Transformer<VectorEmbedding[]> {}

export interface DocumentFragmentVectorEmbedding extends VectorEmbedding {
  fragmentId: string;
}

export abstract class DocumentFragmentEmbeddingsTransformer
  implements EmbeddingsTransformer
{
  /**
   * Embed the given DocumentFragment as a DocumentFragmentVectorEmbedding.
   * @param text The DocumentFragment to embed.
   */
  abstract embedFragment(
    fragment: DocumentFragment
  ): Promise<DocumentFragmentVectorEmbedding>;

  /**
   * Embed all fragments for a single document as DocumentFragmentVectorEmbeddings.
   * This is a convenience method which can be overridden to use the underlying embedding
   * API's batch embedding creation implementation.
   * @param document The document to embed.
   */
  async embedDocument(
    document: Document
  ): Promise<DocumentFragmentVectorEmbedding[]> {
    return await Promise.all(
      document.fragments.map(async (fragment) => this.embedFragment(fragment))
    );
  }

  /**
   * Embed multiple documents as DocumentFragmentVectorEmbeddings. This is a convenience
   * method which can be overridden to use the underlying embedding API's recommended
   * batch embedding creation implementation.
   * @param documents The documents to embed.
   */
  async transformDocuments(
    documents: Document[]
  ): Promise<DocumentFragmentVectorEmbedding[]> {
    const embeddings = await Promise.all(
      documents.map((document) => this.embedDocument(document))
    );
    return embeddings.flat();
  }
}
