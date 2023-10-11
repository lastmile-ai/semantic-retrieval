import { AccessPassport } from "../access-control/accessPassport";
import { Document } from "../document/document";
import { DocumentMetadataDB } from "../document/metadata/documentMetadataDB";

export type BaseRetrieverQueryParams<Q> = {
  accessPassport: AccessPassport;
  query: Q;
};

/**
 * Abstract base class for retrieving data R from an underlying source. Each retriever
 * implementation should override `_getDocumentsUnsafe` to implement their custom retrieval logic
 * and `_processDocuments` to implement desired post-processing logic.
 * The BaseRetriever class provides a `retrieveData` method that wraps `getDocumentsUnsafe` with
 * default access control logic. It is OK to override `retrieveData` if additional access control
 * handling is required (e.g. if the underlying source can perform optimized RBAC), but the quality
 * and correctness of the access control logic is the responsibility of the retriever implementation.
 */
export abstract class BaseRetriever<R> {
  metadataDB: DocumentMetadataDB;

  constructor(metadataDB: DocumentMetadataDB) {
    this.metadataDB = metadataDB;
  }

  /**
   * Get the Documents relevant to the given query without performing any access control checks.
   * @param query The query string to obtain relevant Documents for.
   * @returns A promise that resolves to array of retrieved Documents.
   */
  protected abstract getDocumentsUnsafe<Q>(
    _params: BaseRetrieverQueryParams<Q>
  ): Promise<Document[]>;

  /**
   * Simple filtering of Documents with respect to access policies.
   * @param accessPassport The AccessPassport for the current identity.
   * @param metadataDB The DocumentMetadataDB to use for access control checks.
   * @param documents The Documents to filter.
   * @returns A promise that resolves to array of Documents accessible to the current identity.
   */
  protected async filterAccessibleDocuments(
    accessPassport: AccessPassport,
    documents: Document[]
  ): Promise<Document[]> {
    if (this.metadataDB == null) {
      return documents;
    }

    const accessibleDocuments = await Promise.all(
      documents.map(async (unsafeDocument) => {
        const metadata = await this.metadataDB!.getMetadata(
          unsafeDocument.documentId
        );

        if (metadata.accessPolicies) {
          const policyChecks = await Promise.all(
            metadata.accessPolicies.map(
              async (policy) =>
                await policy.testDocumentReadPermission(
                  unsafeDocument,
                  policy.resource
                    ? accessPassport.getIdentity(policy.resource)
                    : undefined
                )
            )
          );

          if (policyChecks.some((check) => check === false)) {
            return null;
          }
        }
        return unsafeDocument;
      })
    );

    return accessibleDocuments.filter((doc): doc is Document => doc != null);
  }

  /**
   * Perform any post-processing on the retrieved Documents.
   * @param documents The array of retrieved Documents to post-process.
   * @returns A promise that resolves to post-processed data.
   */
  protected abstract processDocuments(_documents: Document[]): Promise<R>;

  /**
   * Get the data relevant to the given query and which the current identity can access.
   * @param params The retriever query params to use for the query.
   * @returns A promise that resolves to the retrieved data.
   */
  async retrieveData<Q>(params: BaseRetrieverQueryParams<Q>): Promise<R> {
    // By default, just perform a single query to the underlying source and filter the results
    // on access control checks, if applicable
    const unsafeDocuments = await this.getDocumentsUnsafe(params);

    const accessibleDocuments = await this.filterAccessibleDocuments(
      params.accessPassport,
      unsafeDocuments
    );

    return await this.processDocuments(accessibleDocuments);
  }
}
