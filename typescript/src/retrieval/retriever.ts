import { AccessPassport } from "../access-control/accessPassport";
import { DocumentMetadataDB } from "../document/metadata/documentMetadataDB";
import { CallbackManager, Traceable } from "../utils/callbacks";

export interface BaseRetrieverQueryParams<Q = unknown> {
  accessPassport?: AccessPassport;
  query: Q;
}

export type RetrieverResponse<R> = R extends BaseRetriever<infer RR>
  ? RR
  : never;

export type RetrieverParams<R> = R extends BaseRetriever<infer _RR, infer RP>
  ? RP
  : never;

export type RetrieverParamsQuery<P> = P extends BaseRetrieverQueryParams<
  infer Q
>
  ? Q
  : never;

export type RetrieverQuery<R> = R extends BaseRetriever<infer _RR, infer RP>
  ? RP extends BaseRetrieverQueryParams<infer QP>
    ? QP
    : never
  : never;

/**
 * Abstract base class for retrieving data R from from an underlying source based on query Q.
 */
export abstract class BaseRetriever<
  R = unknown,
  P extends BaseRetrieverQueryParams = BaseRetrieverQueryParams,
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
