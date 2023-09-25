import { RawDocument } from "../document/document";
import { ResourceAccessPolicy } from "./resourceAccessPolicy";

/**
 * Interface for specifying the access policies for a document during ingestion.
 */
export interface DocumentAccessPolicyFactory {
  /**
   * Specify the access policies for a document during ingestion.
   * @param rawDocument - The RawDocument to specify policies for.
   * @returns list of ResourceAccessPolicies applicable to the document.
   */
  getAccessPolicies: (
    rawDocument: RawDocument,
  ) => Promise<ResourceAccessPolicy[]>;
}
