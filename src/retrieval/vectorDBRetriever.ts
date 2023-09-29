import { VectorDB, VectorDBTextQuery } from "../data-store/vector-DBs/vectorDB";
import { Document } from "../document/document";
import { DocumentMetadataDB } from "../document/metadata/documentMetadataDB";
import { BaseRetriever, BaseRetrieverQueryParams } from "./retriever";

export type VectorDBRetrieverParams<V extends VectorDB> = {
  vectorDB: V;
  metadataDB?: DocumentMetadataDB;
  k?: number;
};

/**
 * Abstract class for retrieving data R from an underlying VectorDB
 */
export abstract class BaseVectorDBRetriever<
  V extends VectorDB,
  R,
> extends BaseRetriever<R> {
  vectorDB: V;

  constructor(vectorDB: V) {
    super(vectorDB.metadataDB);
    this.vectorDB = vectorDB;
  }

  protected async getDocumentsUnsafe(
    params: BaseRetrieverQueryParams,
  ): Promise<Document[]> {
    const query: VectorDBTextQuery = { text: params.query };
    return await this.vectorDB.query(query);
  }
}
