import { VectorDB } from "../data-store/vector-DBs/vectorDB.js";
import { Document } from "../document/document.js";
import { BaseVectorDBRetriever } from "./vectorDBRetriever.js";

/**
 * Base class for retrieving Documents from an underlying VectorDB
 */
export class VectorDBDocumentRetriever<
  V extends VectorDB,
> extends BaseVectorDBRetriever<V, Document[]> {
  constructor(vectorDB: V) {
    super(vectorDB);
  }

  protected async _processDocuments(
    documents: Document[],
  ): Promise<Document[]> {
    return documents;
  }
}
