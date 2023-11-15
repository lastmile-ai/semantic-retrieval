import { JSONObject } from "../common/jsonTypes";
import { Document } from "../document/document";
import { DocumentMetadataDB } from "../document/metadata/documentMetadataDB";
import { AccessIdentity } from "./accessIdentity";

// Resource for globally-scoped access policies (e.g. AlwaysAllowAccessPolicy)
// AccessPassport will always have access to this resource
export const GLOBAL_RESOURCE = "*";

/**
 * Access policy for a resource
 */
export interface ResourceAccessPolicy {
  policy: string;
  resource: string;
  policyJSON?: JSONObject;

  /**
   * Tests whether the requestor has read permission for the specified document containing this policy.
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

  /**
   * Tests whether the user has read permission for this policy.
   * This is used to test whether a resource can be read by the requestor.
   * @param requestor - The identity requesting access to the resource.
   * @returns either a list of permissions that the user has for this scope, or false if the user has no permissions.
   * @see https://cloud.google.com/resource-manager/reference/rest/v3/organizations/testIamPermissions and https://developers.google.com/drive/api/reference/rest/v3/permissions
   */
  testPolicyPermission: (
    requestor: AccessIdentity
  ) => Promise<string[] | boolean>;
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
