import { Document } from "../../document/document";
import { AccessIdentity } from "../accessIdentity";
import { ResourceAccessPolicy } from "../resourceAccessPolicy";

export class AlwaysAllowAccessPolicy implements ResourceAccessPolicy {
  policy: string = "AlwaysAllowAccessPolicy";
  policyJSON: { [key: string]: any } = {};

  async testDocumentReadPermission(
    _document: Document,
    _requestor?: AccessIdentity
  ): Promise<boolean> {
    return true;
  }

  async testPolicyPermission(_requestor: any): Promise<string[] | boolean> {
    return true;
  }
}