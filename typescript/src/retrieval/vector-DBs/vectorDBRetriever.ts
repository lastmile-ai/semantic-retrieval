import { VectorDB, VectorDBQuery } from "../../data-store/vector-DBs/vectorDB";
import {
  DocumentFragment,
  DocumentFragmentType,
} from "../../document/document";
import { DocumentMetadataDB } from "../../document/metadata/documentMetadataDB";
import { BaseRetrieverQueryParams } from "../retriever";
import { DocumentRetriever } from "../documentRetriever";
import { Md5 } from "ts-md5";
import {
  CallbackManager,
  GetFragmentsEvent,
  RetrieveDataEvent,
} from "../../utils/callbacks";

export type VectorDBRetrieverParams = {
  vectorDB: VectorDB;
  metadataDB: DocumentMetadataDB;
  callbackManager?: CallbackManager;
};

export interface VectorDBRetrieverQueryParams
  extends BaseRetrieverQueryParams<VectorDBQuery> {}

/**
 * Abstract class for retrieving data R from an underlying VectorDB
 */
export abstract class BaseVectorDBRetriever<
  R = unknown,
> extends DocumentRetriever<R> {
  vectorDB: VectorDB;

  constructor(params: VectorDBRetrieverParams) {
    super(params.metadataDB, params.callbackManager);
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

    const documentFragments = fragments.filter(
      (fragment) => fragment != null
    ) as DocumentFragment[];

    const event: GetFragmentsEvent = {
      name: "onGetFragments",
      fragments: documentFragments,
    };
    await this.callbackManager?.runCallbacks(event);

    return documentFragments;
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
    const preFilterTopK = Math.min(requestedTopK * 10, 50);
    const unsafeFragments = await this.getFragmentsUnsafe({
      ...params,
      query: { ...params.query, topK: preFilterTopK },
    });

    const accessibleFragments = (
      await this.filterAccessibleFragments(
        unsafeFragments,
        params.accessPassport
      )
    ).slice(0, requestedTopK);

    const accessibleDocuments =
      await this.getDocumentsForFragments(accessibleFragments);

    const processedDocuments = await this.processDocuments(accessibleDocuments);

    const event: RetrieveDataEvent = {
      name: "onRetrieveData",
      params,
      data: processedDocuments,
    };
    await this.callbackManager?.runCallbacks(event);

    return processedDocuments;
  }
}
