import { JSONObject } from "../common/jsonTypes";
import { Document } from "../document/document";
import { AccessIdentity } from "./accessIdentity";

/**
 * Access policy for a resource
 */
export interface ResourceAccessPolicy {
  resource?: string;
  policy: string;
  policyJSON: JSONObject;

  /**
   * Tests whether the requestor has read permission for the specified document.
   * @param document - The document to test access for.
   * @param requestor - The user identity access to the resource.
   * @returns true if the user has read access to the document, false otherwise.
   */
  testDocumentReadPermission: (
    document: Document,
    requestor?: AccessIdentity,
  ) => Promise<boolean>;

  /**
   * Tests whether the user has read permission for this IAM policy.
   * This is used to test whether documents can be read by the requestor.
   * @param requestor - The identity requesting access to the resource.
   * @returns either a list of permissions that the user has for this scope, or false if the user has no permissions.
   * @see https://cloud.google.com/resource-manager/reference/rest/v3/organizations/testIamPermissions and https://developers.google.com/drive/api/reference/rest/v3/permissions
   */
  testPolicyPermission: (
    requestor: AccessIdentity,
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
    requestor: AccessIdentity,
  ): string[] | boolean | undefined {
    return this.cache.get(JSON.stringify(requestor) + policy);
  }

  set(
    policy: string,
    requestor: AccessIdentity,
    permissions: string[] | boolean,
  ) {
    this.cache.set(JSON.stringify(requestor) + policy, permissions);
  }
}
