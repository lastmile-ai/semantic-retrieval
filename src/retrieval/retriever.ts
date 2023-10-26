import { AccessPassport } from "../access-control/accessPassport";
import { DocumentMetadataDB } from "../document/metadata/documentMetadataDB";
import { CallbackManager, Traceable } from "../utils/callbacks";

export type BaseRetrieverQueryParams<Q> = {
  accessPassport?: AccessPassport;
  query: Q;
};

/**
 * Abstract base class for retrieving data R from from an underlying source.
 */
export abstract class BaseRetriever<R, Q> implements Traceable {
  metadataDB: DocumentMetadataDB;
  callbackManager?: CallbackManager;

  constructor(
    metadataDB: DocumentMetadataDB,
    callbackManager?: CallbackManager
  ) {
    this.metadataDB = metadataDB;
    this.callbackManager = callbackManager;
  }

  /**
   * Get the data relevant to the given query and which the current identity can access.
   * @param params The retriever query params to use for the query.
   * @returns A promise that resolves to the retrieved data.
   */
  abstract retrieveData(params: BaseRetrieverQueryParams<Q>): Promise<R>;
}
