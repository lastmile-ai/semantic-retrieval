import { JSONObject } from "../common/jsonTypes";
import { Document } from "../document/document";
import { DocumentMetadataDB } from "../document/metadata/documentMetadataDB";
import { AccessIdentity } from "./accessIdentity";

/**
 * Access policy for a resource
 */
export interface ResourceAccessPolicy {
  policy: string;
  resource?: string;
  policyJSON?: JSONObject;

  /**
   * Tests whether the requestor has read permission for the specified document.
   * @param document - The document to test access for.
   * @param metadataDB - The metadata database to use for access control.
   * @param requestor - The user identity access to the resource.
   * @returns true if the user has read access to the document with respect to this policy, false otherwise.
   */
  testDocumentReadPermission: (
    document: Document,
    metadataDB: DocumentMetadataDB,
    requestor?: AccessIdentity
  ) => Promise<boolean>;
}

/**
 * An in-memory cache of the rest of test*Permission calls.
 */
export class ResourceAccessPolicyCache {
  cache: Map<string, string[] | boolean> = new Map();

  static instance: ResourceAccessPolicyCache = new ResourceAccessPolicyCache();

  get(
    policy: string,
    requestor: AccessIdentity
  ): string[] | boolean | undefined {
    return this.cache.get(JSON.stringify(requestor) + policy);
  }

  set(
    policy: string,
    requestor: AccessIdentity,
    permissions: string[] | boolean
  ) {
    this.cache.set(JSON.stringify(requestor) + policy, permissions);
  }
}
