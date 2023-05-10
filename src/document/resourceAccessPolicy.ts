import { Authentication } from "../ingestion/data-sources/dataSource";

/**
 * Access policy for a resource
 */
export interface ResourceAccessPolicy {
  policy: string;
  policyJSON: { [key: string]: any };

  /**
   * Tests whether the user has read permission for the specified resource.
   * This is used to test whether documents can be read by the requestor.
   * @param resourceId - The resource to test access for.
   * @param requestor - The user requesting access to the resource.
   * @returns true if the user has read access to the resource, false otherwise.
   */
  testResourceReadPermission: (
    resourceId: string,
    requestor: Authentication
  ) => Promise<boolean>;

  /**
   * Tests whether the user has read permission for this IAM policy.
   * This is used to test whether documents can be read by the requestor.
   * @param requestor - The user requesting access to the resource.
   * @returns either a list of permissions that the user has for this scope, or false if the user has no permissions.
   * @see https://cloud.google.com/resource-manager/reference/rest/v3/organizations/testIamPermissions and https://developers.google.com/drive/api/reference/rest/v3/permissions
   */
  testPolicyPermission: (
    requestor: Authentication
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
    requestor: Authentication
  ): string[] | boolean | undefined {
    return this.cache.get(JSON.stringify(requestor) + policy);
  }

  set(
    policy: string,
    requestor: Authentication,
    permissions: string[] | boolean
  ) {
    this.cache.set(JSON.stringify(requestor) + policy, permissions);
  }
}
