import { AccessPassport } from "../access-control/accessPassport";
import { Document, DocumentFragment } from "../document/document";
import { promises as fs } from "fs";
import {
  CallbackManager,
  RetrieveDataEvent,
  RetrieverFilterAccessibleFragmentsEvent,
  RetrieverGetDocumentsForFragmentsEvent,
} from "../utils/callbacks";
import { BaseRetriever, BaseRetrieverQueryParams } from "./retriever";
import { DocumentMetadataDB } from "../document/metadata/documentMetadataDB";

/**
 * Abstract base class for retrieving data R from documents retrieved from an underlying source.
 * Each document retriever implementation should override `getFragmentsUnsafe` to implement their
 * custom retrieval logic and `processDocuments` to implement desired post-processing logic.
 * The DocumentRetriever class provides a `retrieveData` method that wraps `getFragmentsUnsafe` with
 * default access control logic. It is OK to override `retrieveData` if additional access control
 * handling is required (e.g. if the underlying source can perform optimized RBAC), but the quality
 * and correctness of the access control logic is the responsibility of the retriever implementation.
 */
export abstract class DocumentRetriever<R = unknown> extends BaseRetriever {
  metadataDB: DocumentMetadataDB;

  constructor(
    metadataDB: DocumentMetadataDB,
    callbackManager?: CallbackManager
  ) {
    super(metadataDB, callbackManager);
    this.metadataDB = metadataDB;
  }

  /**
   * Get the DocumentFragments relevant to the given query without performing any access control checks.
   * @param query The query string to obtain relevant DocumentFragments for.
   * @returns A promise that resolves to array of retrieved DocumentFragments.
   */
  protected abstract getFragmentsUnsafe(
    _params: BaseRetrieverQueryParams
  ): Promise<DocumentFragment[]>;

  /**
   * Simple filtering of DocumentFragments with respect to access policies.
   * @param fragments The DocumentFragments to filter.
   * @param accessPassport The AccessPassport for the current identity.
   * @returns A promise that resolves to array of DocumentFragments accessible to the current identity.
   */
  protected async filterAccessibleFragments(
    fragments: DocumentFragment[],
    accessPassport?: AccessPassport
  ): Promise<DocumentFragment[]> {
    if (this.metadataDB == null) {
      return fragments;
    }

    const accessibleFragments = await Promise.all(
      fragments.map(async (fragment) => {
        const metadata = await this.metadataDB!.getMetadata(
          fragment.documentId
        );

        // Default to hidden; only show fragments with explicitly-allowing policies
        if (!(metadata && metadata.accessPolicies && metadata.document)) {
          await this.callbackManager?.runCallbacks({
            name: "onRetrievedFragmentPolicyCheckFailed",
            fragment,
            policy: null,
          });
          return null;
        }

        // TODO: Should we force-check source document policies here before the document itself? Since we
        // default to hidden, we would require different policies for the source document and any later
        // documents that don't have access to the rawDocument to check the source (e.g. source document has
        // policy based on source but transformed document has policy not based on source)

        const policyChecks = await Promise.all(
          metadata.accessPolicies.map(async (policy) => ({
            policy,
            passed: await policy.testDocumentReadPermission(
              metadata.document!,
              this.metadataDB!,
              policy.resource
                ? accessPassport?.getIdentity(policy.resource)
                : undefined
            ),
          }))
        );

        for (const check of policyChecks) {
          if (!check.passed) {
            await this.callbackManager?.runCallbacks({
              name: "onRetrievedFragmentPolicyCheckFailed",
              fragment,
              policy: check.policy,
            });

            return null;
          }
        }

        return fragment;
      })
    );

    const filteredFragments = accessibleFragments.filter(
      (fragment): fragment is DocumentFragment => fragment != null
    );

    const event: RetrieverFilterAccessibleFragmentsEvent = {
      name: "onRetrieverFilterAccessibleFragments",
      fragments: filteredFragments,
    };

    await this.callbackManager?.runCallbacks(event);

    return filteredFragments;
  }

  /**
   * Constructs Documents from the given DocumentFragments. Fragments with the same documentId will
   * be grouped into a single Document.
   * @param fragments
   * @returns
   */
  protected async getDocumentsForFragments(
    fragments: DocumentFragment[]
  ): Promise<Document[]> {
    const fragmentsByDocumentId = fragments.reduce(
      (acc, fragment) => {
        if (!fragment) {
          return acc;
        }
        if (acc[fragment.documentId] == null) {
          acc[fragment.documentId] = [];
        }
        acc[fragment.documentId].push(fragment);
        return acc;
      },
      {} as Record<string, DocumentFragment[]>
    );

    // We construct Document objects from the subset of fragments obtained from the query.
    // If needed, all Document fragments can be retrieved from the DocumentMetadataDB.
    const documents = await Promise.all(
      Object.entries(fragmentsByDocumentId).map(
        async ([documentId, fragments]) => {
          const documentMetadata = {
            ...(await this.metadataDB.getMetadata(documentId)),
          };

          const storedDocument = documentMetadata?.document;
          delete documentMetadata.document;

          return {
            ...documentMetadata,
            documentId,
            fragments,
            attributes: documentMetadata?.attributes ?? {},
            metadata: documentMetadata?.metadata ?? {},
            serialize: async () => {
              if (storedDocument) {
                return await storedDocument.serialize();
              }

              const serializedFragments = (
                await Promise.all(
                  fragments.map(async (fragment) => await fragment.serialize())
                )
              ).join("\n");

              const filePath = `${documentId}.txt`;
              await fs.writeFile(filePath, serializedFragments);
              return filePath;
            },
          };
        }
      )
    );

    const event: RetrieverGetDocumentsForFragmentsEvent = {
      name: "onRetrieverGetDocumentsForFragments",
      documents,
    };
    await this.callbackManager?.runCallbacks(event);

    return documents;
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
  async retrieveData(params: BaseRetrieverQueryParams): Promise<R> {
    // By default, just perform a single query to the underlying source and filter the results
    // on access control checks, if applicable
    const unsafeFragments = await this.getFragmentsUnsafe(params);

    const accessibleFragments = await this.filterAccessibleFragments(
      unsafeFragments,
      params.accessPassport
    );

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
