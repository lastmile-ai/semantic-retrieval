import { VectorDB, VectorDBQuery } from "../data-store/vector-DBs/vectorDB";
import {
  DocumentFragment,
  DocumentFragmentType,
} from "../document/document";
import { DocumentMetadataDB } from "../document/metadata/documentMetadataDB";
import { BaseRetriever, BaseRetrieverQueryParams } from "./retriever";
import { Md5 } from "ts-md5";

export type VectorDBRetrieverParams<V extends VectorDB> = {
  vectorDB: V;
  metadataDB: DocumentMetadataDB;
};

export interface VectorDBRetrieverQueryParams
  extends BaseRetrieverQueryParams<VectorDBQuery> {}

/**
 * Abstract class for retrieving data R from an underlying VectorDB
 */
export abstract class BaseVectorDBRetriever<
  V extends VectorDB,
  R,
> extends BaseRetriever<R, VectorDBQuery> {
  vectorDB: V;

  constructor(params: VectorDBRetrieverParams<V>) {
    super(params.metadataDB);
    this.vectorDB = params.vectorDB;
  }

  protected async getFragmentsUnsafe(
    params: VectorDBRetrieverQueryParams
  ): Promise<DocumentFragment[]> {
    const embeddings = await this.vectorDB.query(params.query);
    const fragments: (DocumentFragment | null)[] = await Promise.all(
      embeddings.map(async (embedding) => {
        const { documentId, fragmentId, ...embeddingMetadata } =
          embedding.metadata;

        if (!(fragmentId && documentId)) {
          console.error(
            "Retrieved embedding does not have fragmentId or documentId"
          );
          return null;
        }

        return {
          fragmentId,
          fragmentType: (embeddingMetadata.fragmentType ??
            "text") as DocumentFragmentType,
          documentId,
          metadata: embeddingMetadata,
          attributes: embedding.attributes,
          hash: Md5.hashStr(embedding.text),
          getContent: async () => embedding.text,
          serialize: async () => embedding.text,
        };
      })
    );

    return fragments.filter(
      (fragment) => fragment != null
    ) as DocumentFragment[];
  }

  // Many VectorDBs don't support pagination, so we try to obtain the best K results after
  // filtering on access control checks. We can either query for more than the top K results
  // and filter to the best accessible K, or can query for the top K results and perform
  // successive re-querying until we have K accessible results.
  // The latter requires more specific knowledge of the underlying VectorDB implementation
  // to determine filtering.
  // The default will use the former, fetching up to MIN[K * 10, 50] results and filtering
  // to the top K.
  async retrieveData(params: VectorDBRetrieverQueryParams): Promise<R> {
    const requestedTopK = params.query.topK;
    const preFilterTopK = Math.min(requestedTopK * 10, 500);
    const unsafeFragments = await this.getFragmentsUnsafe({
      ...params,
      query: { ...params.query, topK: preFilterTopK },
    });

    const accessibleFragments = (
      await this.filterAccessibleFragments(
        params.accessPassport,
        unsafeFragments
      )
    ).slice(0, requestedTopK);

    const accessibleDocuments =
      await this.getDocumentsForFragments(accessibleFragments);

    return await this.processDocuments(accessibleDocuments);
  }
}
