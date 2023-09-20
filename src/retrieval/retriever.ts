import { AccessPassport } from "../access-control/accessPassport";
import { Document } from "../document/document";
import { DocumentMetadataDB } from "../document/metadata/documentMetadataDB";

export type BaseRetrieverQueryParams = {
  accessPassport: AccessPassport;
  metadataDB?: DocumentMetadataDB;
  query: string;
};

/**
 * Abstract base class for retrieving Documents from an underlying source. Each retriever
 * implementation should override `_getDocumentsUnsafe` to implement their custom retrieval logic.
 * The BaseRetriever class provides a `getDocuments` method that wraps `getDocumentsUnsafe` with
 * default access control logic. It is OK to override `getDocuments` if additional access control
 * handling is required (e.g. if the underlying source can perform optimized RBAC), but the quality
 * and correctness of the access control logic is the responsibility of the retriever implementation.
 */
export abstract class BaseRetriever {
  constructor() {}

  /**
   * Get the documents relevant to the given query without performing any access control checks.
   * @param query The query string to obtain relevant Documents for.
   * @returns A promise that resolves to array of retrieved Documents.
   */
  protected abstract _getDocumentsUnsafe(
    _params: BaseRetrieverQueryParams
  ): Promise<Document[]>;

  /**
   * Get the documents relevant to the given query and which the current identity can access.
   * @param query The query string to obtain relevant Documents for.
   * @returns A promise that resolves to array of retrieved Documents.
   */
  async getDocuments(params: BaseRetrieverQueryParams): Promise<Document[]> {
    // By default, just perform a single query to the underlying source and filter the results
    // on access control checks, if applicable
    const unsafeDocuments = await this._getDocumentsUnsafe(params);

    const { accessPassport, metadataDB } = params;

    if (!metadataDB) {
      // No metadataDB, so no policies to check for each document
      return unsafeDocuments;
    }

    const safeDocuments = await Promise.all(
      unsafeDocuments.map(async (unsafeDocument) => {
        const metadata = await metadataDB.getMetadata(
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

    return safeDocuments.filter((doc): doc is Document => doc != null);
  }
}
