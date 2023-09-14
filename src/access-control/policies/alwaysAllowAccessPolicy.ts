import { ResourceAccessPolicy } from "../resourceAccessPolicy";

export class AlwaysAllowAccessPolicy implements ResourceAccessPolicy {
  policy: string = "AlwaysAllowAccessPolicy";
  policyJSON: { [key: string]: any } = {};

  async testResourceReadPermission(
    _resourceId: string,
    _requestor: any
  ): Promise<boolean> {
    return true;
  }

  async testPolicyPermission(_requestor: any): Promise<string[] | boolean> {
    return true;
  }
}