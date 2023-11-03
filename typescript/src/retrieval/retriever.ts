import { AccessPassport } from "../access-control/accessPassport";
import { DocumentMetadataDB } from "../document/metadata/documentMetadataDB";
import { CallbackManager, Traceable } from "../utils/callbacks";

export type BaseRetrieverQueryParams<Q> = {
  accessPassport?: AccessPassport;
  query: Q;
};

export type RetrieverParams<R> = R extends BaseRetriever<infer RP, infer _RR>
  ? RP
  : never;

export type RetrieverResponse<R> = R extends BaseRetriever<infer _RQ, infer RR>
  ? RR
  : never;

export type RetrieverQuery<R> = R extends BaseRetriever<
  infer _RP,
  infer _RR,
  infer RQ
>
  ? RQ
  : never;

/**
 * Abstract base class for retrieving data R from from an underlying source based on query Q.
 */
export abstract class BaseRetriever<
  P extends BaseRetrieverQueryParams<Q>,
  R,
  Q = P extends BaseRetrieverQueryParams<infer RQ> ? RQ : never,
> implements Traceable
{
  metadataDB?: DocumentMetadataDB;
  callbackManager?: CallbackManager;

  constructor(
    metadataDB?: DocumentMetadataDB,
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
  abstract retrieveData(params: P): Promise<R>;
}
